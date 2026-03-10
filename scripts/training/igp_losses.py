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
import torch.nn.functional as F
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
    IGP 损失函数
    
    负责计算对比损失 (Ranking Loss) 和辅助损失 (Aux Loss)。
    
    注意: IGP 模块 (Probe/Adapter/Gate) 的调用已移到 Model.forward() 中，
    这里只负责从模型输出中提取结果并计算损失。
    
    流程:
        1. 从 base_model (IGPColBERTWrapper) 获取增强后的 embeddings
        2. 计算对比损失 (Ranking Loss)
        3. 计算辅助损失 (Aux Loss，如果提供了 attn_logits)
        4. 计算正则化损失 (Reg Loss，如果提供了 delta)
        5. 返回总损失
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        base_model: nn.Module,
        probe=None,
        adapter=None,
        gate=None,
        aux_loss_weight: float = 0.1,
        reg_coeff: float = 0.05,
        gate_l1_coeff: float = 0.01,  # L1稀疏正则化系数
    ):
        super().__init__()
        self.base_loss = base_loss
        self.base_model = base_model
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        self.aux_loss_weight = aux_loss_weight
        self.reg_coeff = reg_coeff
        self.gate_l1_coeff = gate_l1_coeff  # L1稀疏正则化系数
        self.aux_loss_fn = IGPAuxLoss()
        self.rank_loss_fn = nn.CrossEntropyLoss()
        self.temperature = 0.05  # ColBERT 通常使用更小的温度
    
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        IGP 损失前向传播
        
        Args:
            sentence_features: 输入特征列表 [query, positive, negative]
            
        Returns:
            total_loss: 总损失 (rank_loss + aux_loss + reg_loss)
        """
        if isinstance(sentence_features, dict):
            # 处理字典格式（来自 collator）
            query_features = {
                'input_ids': sentence_features.get('sentence_0_input_ids'),
                'attention_mask': sentence_features.get('sentence_0_attention_mask'),
                'token_labels': sentence_features.get('sentence_0_token_labels'),
                'instruction_mask': sentence_features.get('sentence_0_instruction_mask'),
            }
            positive_features = {
                'input_ids': sentence_features.get('sentence_1_input_ids'),
                'attention_mask': sentence_features.get('sentence_1_attention_mask'),
            }
            negative_features = {
                'input_ids': sentence_features.get('sentence_2_input_ids'),
                'attention_mask': sentence_features.get('sentence_2_attention_mask'),
            }
        elif isinstance(sentence_features, (list, tuple)) and len(sentence_features) >= 3:
            # 处理列表格式（来自 collect_features）
            # features[0] = query, features[1] = positive, features[2] = negative
            query_features = sentence_features[0]
            positive_features = sentence_features[1]
            negative_features = sentence_features[2]
        else:
            raise ValueError(f"Unsupported sentence_features type: {type(sentence_features)}")
        
        query_attention_mask = query_features.get('attention_mask')
        # token_labels 用于标识 instruction 部分 (1=instruction, 0=query)
        instruction_mask = query_features.get('token_labels')
        
        # ========== 1. 调用 base_model (IGPColBERTWrapper) 获取增强后的 embeddings ==========
        # base_model 的 forward 已经应用了 Probe/Adapter/Gate
        query_result = self.base_model(
            query_input_ids=query_features.get('input_ids'),
            query_attention_mask=query_attention_mask,
            instruction_mask=instruction_mask,
        )
        
        # 从结果中提取增强后的 embeddings 和辅助信息
        query_embeddings = query_result['token_embeddings']  # 已经应用了 IGP 模块
        attn_logits = query_result.get('attn_logits')  # 用于 aux_loss
        gate_ratio = query_result.get('gate_ratio', 0.0)  # 用于日志
        gate_penalty = query_result.get('gate_penalty', torch.tensor(0.0, device=rank_loss.device))  # L1稀疏惩罚
        
        # 获取正负样本 embeddings (不使用 IGP，直接走 base_model[0])
        positive_embeddings = self._get_embeddings(positive_features)
        negative_embeddings = self._get_embeddings(negative_features)
        
        # ========== 2. 计算对比损失 (Ranking Loss) ==========
        # 使用 ColBERT 的 Late Interaction (MaxSim) 机制
        # 归一化
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=-1)
        
        # ColBERT Late Interaction: 对每个 query token，找到最相似的 doc token
        # 然后求和得到最终分数
        def colbert_score(Q, D):
            # Q: [batch, q_len, dim], D: [batch, d_len, dim]
            # 计算所有 query-doc token 对的相似度
            sim = torch.matmul(Q, D.transpose(-2, -1))  # [batch, q_len, d_len]
            # 对每个 query token，取最大相似度
            max_sim = sim.max(dim=-1)[0]  # [batch, q_len]
            # 求和得到最终分数
            return max_sim.sum(dim=-1)  # [batch]
        
        # 计算正负样本分数
        pos_score = colbert_score(query_embeddings, positive_embeddings)
        neg_score = colbert_score(query_embeddings, negative_embeddings)
        
        # 拼接分数用于 CrossEntropy
        scores = torch.stack([pos_score, neg_score], dim=-1)  # [batch, 2]
        
        # CrossEntropy Loss (正样本的 label 是 0)
        rank_labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        rank_loss = F.cross_entropy(scores / self.temperature, rank_labels)
        
        # ========== 3. 计算辅助损失 (Aux Loss) ==========
        # 如果 aux_loss_weight 为 0，跳过 Aux Loss 计算
        aux_loss = torch.tensor(0.0, device=rank_loss.device)
        inst_vec = query_result.get('inst_vec')
        
        if self.aux_loss_weight > 0:
            aux_loss = torch.tensor(0.0, device=rank_loss.device, requires_grad=True)
            
            if attn_logits is not None and instruction_mask is not None and query_attention_mask is not None:
                # 对齐 mask 长度
                seq_len = attn_logits.shape[1]
                if instruction_mask.shape[1] > seq_len:
                    target_mask = instruction_mask[:, 1:1+seq_len]
                else:
                    target_mask = instruction_mask[:, :seq_len]
                target_mask = target_mask[:, :seq_len].float()
                
                aux_loss = self.aux_loss_fn.compute(
                    attn_logits,
                    target_mask,
                    query_attention_mask
                )
        
        # ========== 4. 计算正则化损失 (Reg Loss) ==========
        # 约束 delta 范数，防止数值爆炸
        reg_loss = torch.tensor(0.0, device=rank_loss.device, requires_grad=True)
        # 注意: delta 的计算在 Model.forward 中，这里可以通过梯度惩罚实现
        # 或者通过 L2 正则化约束 query_embeddings 的变化
        
        # ========== 5. 计算门控L1稀疏正则化损失 ==========
        # 使用 wrapper 中计算的 gate_penalty (L1稀疏惩罚)
        # gate_penalty 已经是当前 batch 中 current_ratio 绝对值的平均值
        if isinstance(gate_penalty, torch.Tensor):
            gate_l1_loss = gate_penalty
        else:
            gate_l1_loss = torch.tensor(0.0, device=rank_loss.device)
        
        # ========== 6. 计算总损失 ==========
        # 添加虚拟损失确保梯度流动到 IGP 模块
        # 无论 aux_loss_weight 是否为 0，都需要确保梯度能够流动
        dummy_loss = torch.tensor(0.0, device=rank_loss.device)
        if inst_vec is not None:
            dummy_loss = torch.norm(inst_vec, p=2).mean() * 0.01
        
        # 总损失 = Rank_Loss + 0.01 * gate_penalty
        # gate_penalty 来自 wrapper 的 L1 稀疏惩罚
        total_loss = (rank_loss + 
                     aux_loss * self.aux_loss_weight + 
                     reg_loss * self.reg_coeff + 
                     gate_l1_loss * self.gate_l1_coeff +  # L1稀疏正则
                     dummy_loss)
        
        # 保存各项损失用于日志
        # 将 gate_ratio 转换为 float（如果是 tensor）
        if isinstance(gate_ratio, torch.Tensor):
            # gate_ratio 可能是 [batch_size] 或标量，统一取平均
            gate_ratio_val = gate_ratio.mean().item()
        else:
            gate_ratio_val = gate_ratio
        self._last_losses = {
            'rank_loss': rank_loss.item(),
            'aux_loss': aux_loss.item(),
            'reg_loss': reg_loss.item(),
            'gate_l1_loss': gate_l1_loss.item(),  # L1稀疏正则损失
            'gate_ratio': gate_ratio_val,
            'total_loss': total_loss.item(),
        }
        
        return total_loss
    
    def _get_embeddings(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """获取 embeddings (不使用 IGP，直接走 base_model[0] 并投影到 128 维)"""
        # 对于正负样本，不使用 IGP 模块，直接获取 embeddings
        result = self.base_model.base_model[0](features)
        embeddings = result["token_embeddings"]  # 768维
        
        # 投影到 128 维 (使用 base_model[1] - Dense 层)
        features_dict = {"token_embeddings": embeddings}
        projected = self.base_model.base_model[1](features_dict)
        embeddings_128 = projected["token_embeddings"]
        
        return embeddings_128
    
    def get_last_losses(self) -> dict:
        """获取上一次前向传播的各项损失值"""
        return getattr(self, '_last_losses', {})


class IGPLossPhase1(nn.Module):
    """
    Phase 1 损失函数：仅使用辅助损失
    
    仅训练 Probe 识别指令的能力，不进行对比学习。
    冻结其他所有参数。
    
    注意: 使用 IGPColBERTWrapper 作为 base_model，
    Probe 的调用在 Model.forward 中完成。
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        probe,
        aux_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model  # IGPColBERTWrapper
        self.probe = probe
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_fn = IGPAuxLoss()
    
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Phase 1 前向传播
        
        Args:
            sentence_features: 输入特征列表
            labels: 标签 (不使用)
            
        Returns:
            total_loss: 辅助损失
        """
        if isinstance(sentence_features, dict):
            query_features = {
                'input_ids': sentence_features.get('input_ids'),
                'attention_mask': sentence_features.get('attention_mask'),
                'token_labels': sentence_features.get('token_labels'),
                'instruction_mask': sentence_features.get('instruction_mask'),
            }
        else:
            query_features = sentence_features[0]
        
        query_attention_mask = query_features.get('attention_mask')
        instruction_mask = query_features.get('instruction_mask')
        
        if instruction_mask is None:
            instruction_mask = query_features.get('token_labels')
        
        if instruction_mask is None:
            instruction_mask = query_features.get('labels')
        
        # 调用 base_model (IGPColBERTWrapper) 获取结果
        # 它会调用 Probe 提取指令向量
        result = self.base_model(
            query_input_ids=query_features.get('input_ids'),
            query_attention_mask=query_attention_mask,
            instruction_mask=instruction_mask,
        )
        
        # 从结果中提取 attn_logits (用于 aux_loss)
        attn_logits = result.get('attn_logits')
        
        # 计算辅助损失
        device = query_attention_mask.device if query_attention_mask is not None else torch.device('cpu')
        
        # 如果 aux_loss_weight 为 0，返回 0 损失（不训练 Probe）
        if self.aux_loss_weight <= 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        aux_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 获取 inst_vec 用于虚拟损失（确保梯度流动）
        inst_vec = result.get('inst_vec')
        
        if attn_logits is not None and instruction_mask is not None and query_attention_mask is not None:
            # 对齐 mask 长度
            seq_len = attn_logits.shape[1]
            if instruction_mask.shape[1] > seq_len:
                target_mask = instruction_mask[:, 1:1+seq_len]
            else:
                target_mask = instruction_mask[:, :seq_len]
            target_mask = target_mask[:, :seq_len].float()
            
            aux_loss = self.aux_loss_fn.compute(
                attn_logits,
                target_mask,
                query_attention_mask
            )
        
        # 确保损失连接到计算图（即使 aux_loss 为 0）
        # 使用 inst_vec 的 L2 范数作为虚拟损失，确保 probe 有梯度
        if inst_vec is not None:
            dummy_loss = torch.norm(inst_vec, p=2).mean() * 0.01
            total_loss = (aux_loss + dummy_loss) * self.aux_loss_weight
        else:
            total_loss = aux_loss * self.aux_loss_weight
        
        return total_loss


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
        self.temperature = 0.1
    
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
        
        query_emb = final_embeddings.mean(dim=1)
        pos_emb = positive_embeddings.mean(dim=1)
        neg_emb = negative_embeddings.mean(dim=1)
        
        scores = torch.cat([
            torch.sum(query_emb * pos_emb, dim=-1, keepdim=True),
            torch.sum(query_emb * neg_emb, dim=-1, keepdim=True),
        ], dim=-1)
        
        labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        rank_loss = F.cross_entropy(scores / self.temperature, labels)
        
        if instruction_mask is not None and query_attention_mask is not None:
            aux_loss = self.aux_loss_fn.compute(
                attn_logits, 
                instruction_mask,
                query_attention_mask
            )
        else:
            aux_loss = torch.tensor(0.0, device=rank_loss.device)
        
        total_loss = rank_loss + aux_loss * self.aux_loss_weight
        
        return total_loss


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
