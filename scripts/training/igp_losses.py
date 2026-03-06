"""
IGP 损失函数模块

提供 Phase 1 和 Phase 2 不同的损失计算逻辑。

Phase 1 (Probe Warm-up):
    - 仅使用 Aux Loss (BCE Loss for instruction detection)
    - 冻结除 Probe 外的所有参数
    
Phase 2 (Joint Training):
    - 使用 Rank Loss (对比损失) + Aux Loss
    - 联合训练所有参数
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Iterable


class IGPAuxLoss:
    """
    IGP 辅助损失：指令检测损失 (BCE Loss)
    
    教 Probe 学会从 query 中识别指令词。
    
    技术规范:
        - 只在有效 token 上计算 loss（排除 padding）
        - 使用 pos_weight=10.0 对抗样本不平衡
        - padding 位置不参与任何计算
    
    损失计算逻辑:
        1. 有效位置 = attention_mask == 1
        2. 只在有效位置计算 BCE Loss
        3. padding 位置 loss 置为 0
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
    ):
        self.pos_weight = pos_weight
    
    def compute(
        self,
        attn_logits: torch.Tensor,
        instruction_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算辅助损失
        
        Args:
            attn_logits: 未归一化的注意力分数 [batch_size, seq_len]
            instruction_mask: 指令掩码 [batch_size, seq_len], 0=查询词, 1=指令词
            attention_mask: ColBERT 注意力掩码 [batch_size, seq_len], 1=有效token, 0=padding
            
        Returns:
            aux_loss: 标量损失值
        """
        if instruction_mask is None or attention_mask is None:
            return torch.tensor(0.0, device=attn_logits.device, requires_grad=True)
        
        batch_size, seq_len = attn_logits.shape
        
        target = instruction_mask.float()
        
        pos_weight = torch.ones_like(target) * self.pos_weight
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        
        loss = loss_fn(attn_logits, target)
        
        valid_mask = attention_mask.float()
        
        loss = loss * valid_mask
        
        num_valid = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        
        loss = loss.sum(dim=1, keepdim=True) / num_valid
        
        return loss.mean()
    
    def regularization(
        self,
        delta: torch.Tensor,
        max_norm: float = 1.0,
    ) -> torch.Tensor:
        """
        正则化损失：约束 delta 范数，防止数值爆炸
        
        Args:
            delta: 偏移向量 [batch_size, seq_len, hidden_size]
            max_norm: 最大范数值
            
        Returns:
            reg_loss: 正则化损失
        """
        # 计算 delta 的 L2 范数
        delta_norm = torch.norm(delta, p=2, dim=-1)  # [batch_size, seq_len]
        
        # 超过 max_norm 的部分进行惩罚
        excess = torch.clamp(delta_norm - max_norm, min=0)
        
        return excess.mean()


class IGPAdapterLoss:
    """
    IGP Adapter 专用损失
    
    计算 Adapter 输出的 delta 向量的正则化损失
    """
    
    def __init__(
        self,
        delta_max_norm: float = 1.0,
        delta_weight: float = 0.01,
    ):
        self.delta_max_norm = delta_max_norm
        self.delta_weight = delta_weight
    
    def compute(
        self,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Adapter 正则化损失
        
        Args:
            delta: 偏移向量 [batch_size, seq_len, hidden_size]
            
        Returns:
            reg_loss: 正则化损失
        """
        # L2 正则化：限制 delta 范数
        delta_norm = torch.norm(delta, p=2, dim=-1)  # [batch_size, seq_len]
        
        # 惩罚超过阈值的部分
        excess = torch.clamp(delta_norm - self.delta_max_norm, min=0)
        
        return self.delta_weight * excess.mean()


class IGPLoss(nn.Module):
    """
    IGP 组合损失
    
    集成 Probe、Adapter、Gate 到对比损失计算中。
    
    流程:
        1. 获取 query_embeddings
        2. Probe 生成 instruction_vector
        3. Adapter 注入指令知识
        4. Gate 融合原始和增强表示
        5. 使用增强后的 embeddings 计算对比损失
        6. 计算 Aux Loss (仅当 instruction_mask 可用)
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        base_model: nn.Module,
        probe,
        adapter,
        gate,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.base_model = base_model
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_fn = IGPAuxLoss()
    
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        IGP 损失前向传播
        
        Args:
            sentence_features: [query_features, positive_features, negative_features]
            labels: 标签
            
        Returns:
            (total_loss, info_dict)
        """
        query_features = sentence_features[0]
        positive_features = sentence_features[1]
        negative_features = sentence_features[2]
        
        query_embeddings = self._get_embeddings(query_features)
        positive_embeddings = self._get_embeddings(positive_features)
        negative_embeddings = self._get_embeddings(negative_features)
        
        token_labels = query_features.get('token_labels')
        query_attention_mask = query_features.get('attention_mask')
        
        inst_vec = None
        attn_logits = None
        attn_weights = None
        gate_ratio = None
        adapter_output = None
        
        instruction_mask = query_features.get('instruction_mask')
        
        if self.probe is not None:
            inst_vec, attn_logits, attn_weights = self.probe(
                query_embeddings,
                query_attention_mask
            )
        
        if self.adapter is not None and inst_vec is not None:
            adapter_output = self.adapter(
                query_embeddings,
                instruction_vector=inst_vec
            )
            enhanced_embeddings = adapter_output
        else:
            enhanced_embeddings = query_embeddings
        
        if self.gate is not None and inst_vec is not None:
            fused_embeddings, gate_ratio = self.gate(
                original_vec=query_embeddings,
                instruction_vec=inst_vec.unsqueeze(1).expand(-1, query_embeddings.size(1), -1) if inst_vec.dim() == 2 else inst_vec
            )
            final_embeddings = fused_embeddings
        else:
            final_embeddings = enhanced_embeddings
        
        final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=-1)
        positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=-1)
        
        sentence_features_enhanced = [
            {'token_embeddings': final_embeddings, 'attention_mask': query_attention_mask},
            {'token_embeddings': positive_embeddings, 'attention_mask': positive_features.get('attention_mask')},
            {'token_embeddings': negative_embeddings, 'attention_mask': negative_features.get('attention_mask')},
        ]
        
        rank_loss = self.base_loss.base_loss.compute_score(
            sentence_features_enhanced,
            temperature=self.base_loss.temperature if hasattr(self.base_loss, 'temperature') else 1.0,
        )
        
        if hasattr(self.base_loss, 'gather_across_devices') and self.base_loss.gather_across_devices:
            batch_size = final_embeddings.size(0)
            rank_loss = rank_loss.view(batch_size, -1).mean(-1)
        
        if rank_loss.dim() > 0:
            rank_loss = rank_loss.mean()
        
        aux_loss = torch.tensor(0.0, device=rank_loss.device)
        if attn_logits is not None and instruction_mask is not None and query_attention_mask is not None:
            aux_loss = self.aux_loss_fn.compute(
                attn_logits,
                instruction_mask,
                query_attention_mask
            )
        
        total_loss = rank_loss + aux_loss * self.aux_loss_weight
        
        info = {
            'rank_loss': rank_loss.item() if torch.is_tensor(rank_loss) else rank_loss,
            'aux_loss': aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'gate_ratio': gate_ratio.item() if gate_ratio is not None and torch.is_tensor(gate_ratio) else (gate_ratio if gate_ratio else 0),
            'inst_vec_mean': inst_vec.mean().item() if inst_vec is not None and torch.is_tensor(inst_vec) else 0,
            'attn_weights_mean': attn_weights.mean().item() if attn_weights is not None and torch.is_tensor(attn_weights) else 0,
        }
        
        return total_loss, info
    
    def _get_embeddings(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """获取 embeddings"""
        result = self.base_model(features)
        embeddings = result["token_embeddings"]
        return embeddings


class IGPLossPhase1(nn.Module):
    """
    Phase 1 损失函数：仅使用辅助损失
    
    仅训练 Probe 识别指令的能力，不进行对比学习。
    冻结其他所有参数。
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        probe,
        aux_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.probe = probe
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_fn = IGPAuxLoss()
    
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Phase 1 前向传播
        
        Args:
            sentence_features: ColBERT 输出的特征列表
            labels: 标签 (不使用)
            
        Returns:
            (loss, info_dict)
        """
        query_features = sentence_features[0]
        
        query_embeddings = self.base_model(query_features)["token_embeddings"]
        query_attention_mask = query_features.get('attention_mask')
        
        instruction_mask = query_features.get('instruction_mask')
        
        if instruction_mask is None:
            instruction_mask = query_features.get('token_labels')
        
        if instruction_mask is None:
            instruction_mask = query_features.get('labels')
        
        if instruction_mask is None:
            input_ids = query_features.get('input_ids')
            if input_ids is not None and query_attention_mask is not None:
                batch_size, seq_len = input_ids.shape
                sep_id = self.base_model.tokenizer.sep_token_id
                
                instruction_mask = torch.zeros(batch_size, seq_len, dtype=torch.float, device=input_ids.device)
                
                for i in range(batch_size):
                    sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
                    
                    if len(sep_positions) > 0:
                        first_sep = sep_positions[0].item()
                        instruction_mask[i, first_sep + 1:] = 1.0
        
        # 确保 query_embeddings 参与梯度计算
        query_embeddings = query_embeddings.float()
        
        inst_vec, attn_logits, attn_weights = self.probe(
            query_embeddings,
            query_attention_mask
        )
        
        if instruction_mask is not None and query_attention_mask is not None:
            aux_loss = self.aux_loss_fn.compute(
                attn_logits, 
                instruction_mask, 
                query_attention_mask
            )
        else:
            aux_loss = torch.tensor(0.0, device=query_embeddings.device, requires_grad=True)
        
        aux_loss_weight = torch.tensor(self.aux_loss_weight, device=aux_loss.device, dtype=aux_loss.dtype)
        total_loss = aux_loss * aux_loss_weight
        
        info = {
            'aux_loss': aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
            'inst_vec_mean': inst_vec.mean().item() if torch.is_tensor(inst_vec) else 0,
            'attn_weights_mean': attn_weights.mean().item() if torch.is_tensor(attn_weights) else 0,
            'attn_weights_max': attn_weights.max().item() if torch.is_tensor(attn_weights) else 0,
            'attn_weights_min': attn_weights.min().item() if torch.is_tensor(attn_weights) else 0,
        }
        
        return total_loss, info


class IGPLossPhase2(nn.Module):
    """
    Phase 2 损失函数：对比损失 + 辅助损失
    
    联合训练 Probe、Adapter、Gate 和基础模型。
    总 Loss = Rank Loss + Aux Loss
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        base_model: nn.Module,
        probe,
        adapter,
        gate,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.base_model = base_model
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_fn = IGPAuxLoss()
    
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Phase 2 前向传播
        
        Args:
            sentence_features: ColBERT 输出的特征列表 [query, positive, negative]
            labels: 标签
            
        Returns:
            (loss, info_dict)
        """
        query_features = sentence_features[0]
        positive_features = sentence_features[1]
        negative_features = sentence_features[2]
        
        query_embeddings = self.base_model(query_features)["token_embeddings"]
        positive_embeddings = self.base_model(positive_features)["token_embeddings"]
        negative_embeddings = self.base_model(negative_features)["token_embeddings"]
        
        query_attention_mask = query_features.get('attention_mask')
        instruction_mask = query_features.get('instruction_mask')
        
        if instruction_mask is None:
            instruction_mask = query_features.get('token_labels')
        
        inst_vec, attn_logits, attn_weights = self.probe(
            query_embeddings,
            query_attention_mask
        )
        
        enhanced_embeddings = query_embeddings
        if self.adapter is not None and inst_vec is not None:
            enhanced_embeddings = self.adapter(
                query_embeddings,
                instruction_vector=inst_vec
            )
        
        if self.gate is not None and inst_vec is not None:
            if inst_vec.dim() == 2:
                inst_vec_expanded = inst_vec.unsqueeze(1).expand(-1, query_embeddings.size(1), -1)
            else:
                inst_vec_expanded = inst_vec
            fused_embeddings, gate_ratio = self.gate(
                original_vec=query_embeddings,
                instruction_vec=inst_vec_expanded
            )
            final_embeddings = fused_embeddings
        else:
            final_embeddings = enhanced_embeddings
            gate_ratio = None
        
        final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=-1)
        positive_embeddings = torch.nn.functional.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = torch.nn.functional.normalize(negative_embeddings, p=2, dim=-1)
        
        sentence_features_enhanced = [
            {'token_embeddings': final_embeddings, 'attention_mask': query_attention_mask},
            {'token_embeddings': positive_embeddings, 'attention_mask': positive_features.get('attention_mask')},
            {'token_embeddings': negative_embeddings, 'attention_mask': negative_features.get('attention_mask')},
        ]
        
        rank_loss = self.base_loss.base_loss.compute_score(
            sentence_features_enhanced,
            temperature=self.base_loss.temperature if hasattr(self.base_loss, 'temperature') else 1.0,
        )
        
        if hasattr(self.base_loss, 'gather_across_devices') and self.base_loss.gather_across_devices:
            batch_size = final_embeddings.size(0)
            rank_loss = rank_loss.view(batch_size, -1).mean(-1)
        
        if rank_loss.dim() > 0:
            rank_loss = rank_loss.mean()
        
        if instruction_mask is not None and query_attention_mask is not None:
            aux_loss = self.aux_loss_fn.compute(
                attn_logits, 
                instruction_mask,
                query_attention_mask
            )
        else:
            aux_loss = torch.tensor(0.0, device=rank_loss.device)
        
        total_loss = rank_loss + aux_loss * self.aux_loss_weight
        
        info = {
            'rank_loss': rank_loss.item() if torch.is_tensor(rank_loss) else rank_loss,
            'aux_loss': aux_loss.item() if torch.is_tensor(aux_loss) else aux_loss,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'gate_ratio': gate_ratio.item() if gate_ratio is not None and torch.is_tensor(gate_ratio) else (gate_ratio if gate_ratio else 0),
            'inst_vec_mean': inst_vec.mean().item() if torch.is_tensor(inst_vec) else 0,
            'attn_weights_mean': attn_weights.mean().item() if torch.is_tensor(attn_weights) else 0,
        }
        
        return total_loss, info


class IGPLossSimple(nn.Module):
    """
    简化的 IGP 损失（当前使用的版本）
    
    仅返回 base_loss，aux_loss 未实现
    """
    
    def __init__(
        self,
        base_loss,
    ):
        super().__init__()
        self.base_loss = base_loss
    
    def forward(
        self,
        sentence_features: list,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base_loss(sentence_features, labels)
