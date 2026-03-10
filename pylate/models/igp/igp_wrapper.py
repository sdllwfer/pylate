"""
IGP ColBERT Wrapper - 可复用的 IGP 包装器

将 IGP 模块 (Probe/Adapter/Gate) 集成到 ColBERT 模型中。
可以被训练脚本和评估脚本共同使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IGPColBERTWrapper(nn.Module):
    """
    IGP 包装器：集成 Adapter 和 Gate 到 ColBERT 模型
    
    在 ColBERT 的 token embeddings 输出后注入 instruction 信息：
    1. 使用 Probe 从 Query 中提取指令向量
    2. 使用 Adapter 计算指令偏移量 (delta)
    3. 使用 Gate 控制偏移量的应用强度
    4. 最终生成增强后的 Query Embeddings
    
    注意：此实现与训练脚本中的 forward 逻辑完全一致
    """
    
    def __init__(
        self,
        base_model,
        probe=None,
        adapter=None,
        gate=None,
        freeze_base: bool = True,
        freeze_probe: bool = False,
        freeze_adapter: bool = False,
        freeze_gate: bool = False,
    ):
        """
        Args:
            base_model: 基础 ColBERT 模型 (应该是 Sequential 包含 [encoder, dense])
            probe: InstructionProbe 实例 (可选)
            adapter: IGPAdapter 实例 (可选)
            gate: RatioGate/RatioGateV2/RatioGateV3 实例 (可选)
            freeze_base: 是否冻结基础模型
            freeze_probe: 是否冻结 Probe
            freeze_adapter: 是否冻结 Adapter
            freeze_gate: 是否冻结 Gate
        """
        super().__init__()
        self.base_model = base_model
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        
        # 将 IGP 模块移动到与 base_model 相同的设备
        device = next(base_model.parameters()).device
        if self.probe is not None:
            self.probe = self.probe.to(device)
        if self.adapter is not None:
            self.adapter = self.adapter.to(device)
        if self.gate is not None:
            self.gate = self.gate.to(device)
        
        # 确保 base_model 也在正确的设备上
        self.base_model = self.base_model.to(device)
        
        # 控制各模块的梯度
        self._set_gradients(
            freeze_base=freeze_base,
            freeze_probe=freeze_probe,
            freeze_adapter=freeze_adapter,
            freeze_gate=freeze_gate,
        )
    
    def _set_gradients(
        self,
        freeze_base: bool = True,
        freeze_probe: bool = False,
        freeze_adapter: bool = False,
        freeze_gate: bool = False,
    ):
        """设置各模块的梯度状态"""
        # Base model
        for param in self.base_model.parameters():
            param.requires_grad = not freeze_base
        
        # Probe
        if self.probe is not None:
            for param in self.probe.parameters():
                param.requires_grad = not freeze_probe
        
        # Adapter
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = not freeze_adapter
        
        # Gate
        if self.gate is not None:
            for param in self.gate.parameters():
                param.requires_grad = not freeze_gate
    
    def set_phase1_mode(self):
        """Phase 1: 只训练 Probe，冻结其他所有参数"""
        self._set_gradients(
            freeze_base=True,
            freeze_probe=False,  # 只训练 Probe
            freeze_adapter=True,
            freeze_gate=True,
        )
    
    def set_phase2_mode(self):
        """Phase 2: 联合训练所有 IGP 模块，冻结 Base"""
        self._set_gradients(
            freeze_base=True,    # Base 始终冻结
            freeze_probe=False,  # 训练 Probe
            freeze_adapter=False,  # 训练 Adapter
            freeze_gate=False,   # 训练 Gate
        )
    
    def forward(
        self, 
        query_input_ids=None, 
        query_attention_mask=None,
        pos_doc_input_ids=None,
        pos_doc_attention_mask=None,
        neg_doc_input_ids=None,
        neg_doc_attention_mask=None,
        instruction_mask=None,
        **kwargs
    ):
        """
        前向传播：应用 IGP 模块生成增强的 Query Embeddings
        
        流程:
        1. 获取 Query 的原始 embeddings (768维)
        2. Probe 提取指令向量 inst_vec
        3. Adapter 计算 delta (偏移量)
        4. Gate 控制偏移强度，生成最终 Q_final
        5. 归一化并返回
        
        注意：此逻辑与训练脚本完全一致
        """
        # 如果没有 IGP 模块，直接返回 base_model 的输出
        if self.probe is None and self.adapter is None and self.gate is None:
            return self.base_model(
                query_input_ids=query_input_ids,
                query_attention_mask=query_attention_mask,
                pos_doc_input_ids=pos_doc_input_ids,
                pos_doc_attention_mask=pos_doc_attention_mask,
                neg_doc_input_ids=neg_doc_input_ids,
                neg_doc_attention_mask=neg_doc_attention_mask,
                **kwargs
            )
        
        # ========== 1. 获取 Query 的原始 embeddings (768维) ==========
        # 使用 base_model[0] (Transformer 层) 获取 embeddings，而不是投影后的
        query_features = {'input_ids': query_input_ids, 'attention_mask': query_attention_mask}
        query_out = self.base_model[0](query_features)
        Q_hidden = query_out['token_embeddings']  # [batch, seq, 768]
        
        # ========== 2. Probe: 提取指令向量 (在 768 维空间操作) ==========
        inst_vec = None
        attn_logits = None
        if self.probe is not None:
            probe_output = self.probe(Q_hidden, query_attention_mask)
            if isinstance(probe_output, tuple):
                # 支持返回 2 个或 3 个值的情况
                # InstructionProbe: (inst_vec, attn_logits)
                # InstructionProbeV2: (inst_vec, attn_logits, attn_weights)
                inst_vec = probe_output[0]
                attn_logits = probe_output[1] if len(probe_output) > 1 else None
            else:
                inst_vec = probe_output
        
        # ========== 3. Adapter: 计算 delta (在 768 维空间操作) ==========
        adapter_output = None
        delta = None
        Q_hat = Q_hidden
        if self.adapter is not None and inst_vec is not None:
            # 使用 IGPAdapter.forward 方法，包含 layer_norm 和残差连接
            # concat_dim="hidden" 表示在 hidden 维度拼接，每个 token 都能看到指令向量
            adapter_result = self.adapter(Q_hidden, inst_vec, concat_dim="hidden")
            if isinstance(adapter_result, tuple):
                adapter_output, delta = adapter_result
            else:
                adapter_output = adapter_result
                delta = adapter_output - Q_hidden
        
        # ========== 4. Gate: 全局感知门控 (在 768 维空间操作) ==========
        gate_ratio = torch.tensor(0.0, device=Q_hidden.device)
        gate_penalty = torch.tensor(0.0, device=Q_hidden.device)
        
        if self.gate is not None and inst_vec is not None:
            # 计算 Query 的全局表示: [batch, seq, dim] -> [batch, dim]
            Q_global = Q_hidden.mean(dim=1)
            
            # 使用门控网络根据 Q_global 预测门控比例
            # gate_logits: [batch, 1]
            gate_logits = self.gate(Q_global)
            
            # 硬约束: current_ratio = max_ratio * sigmoid(gate_logits)
            # max_ratio 默认为 0.2
            max_ratio = getattr(self.gate, 'max_ratio', 0.2)
            current_ratio = max_ratio * torch.sigmoid(gate_logits)  # [batch, 1]
            
            # 保存门控比例用于监控
            gate_ratio = current_ratio.squeeze(-1)  # [batch]
            
            # 计算 L1 稀疏惩罚项 (gate_penalty)
            gate_penalty = current_ratio.abs().mean()  # 标量，保留梯度
            
            # 执行加权融合: Q_hat = Q_origin + current_ratio * inst_vec
            # current_ratio: [batch, 1] -> [batch, 1, 1]
            # inst_vec: [batch, dim] -> [batch, 1, dim]
            current_ratio_expanded = current_ratio.unsqueeze(-1)  # [batch, 1, 1]
            inst_vec_expanded = inst_vec.unsqueeze(1)  # [batch, 1, dim]
            
            Q_hat = Q_hidden + current_ratio_expanded * inst_vec_expanded
        elif delta is not None:
            # 当没有 Gate 但有 delta 时，直接使用 adapter_output
            Q_hat = adapter_output if adapter_output is not None else Q_hidden + delta
        else:
            Q_hat = Q_hidden
        
        # ========== 5. 投影到 ColBERT 维度并归一化 (768维 -> 128维) ==========
        # 使用 base_model[1] (Dense 层) 进行投影
        # Dense 层期望字典输入
        features = {"token_embeddings": Q_hat}
        projected_features = self.base_model[1](features)
        Q_projected = projected_features["token_embeddings"]
        
        # ColBERT 必须的 L2 归一化
        Q_final = F.normalize(Q_projected, p=2, dim=-1)
        
        # 计算调试信息
        debug_stats = {}
        if self.probe is not None and inst_vec is not None:
            # 指令向量范数
            debug_stats['inst_vec_norm'] = torch.norm(inst_vec, p=2, dim=-1).mean().item()
        
        if delta is not None:
            # Delta 范数（偏移量大小）
            debug_stats['delta_norm'] = torch.norm(delta, p=2, dim=-1).mean().item()
            # 原始 Query 范数
            debug_stats['Q_hidden_norm'] = torch.norm(Q_hidden, p=2, dim=-1).mean().item()
            # 增强后 Query 范数
            debug_stats['Q_hat_norm'] = torch.norm(Q_hat, p=2, dim=-1).mean().item()
            # 范数变化比例
            debug_stats['norm_change_ratio'] = (debug_stats['Q_hat_norm'] / (debug_stats['Q_hidden_norm'] + 1e-8) - 1) * 100
        
        if isinstance(gate_ratio, torch.Tensor):
            # gate_ratio 可能是 [batch_size] 或标量，统一取平均
            debug_stats['gate_ratio'] = gate_ratio.mean().item()
        else:
            debug_stats['gate_ratio'] = gate_ratio
        
        # 返回结果字典，包含增强后的 embeddings 和辅助信息
        result = {
            'token_embeddings': Q_final,
            'inst_vec': inst_vec,
            'attn_logits': attn_logits,
            'gate_ratio': gate_ratio,
            'gate_penalty': gate_penalty,
            'debug_stats': debug_stats,
        }
        
        return result
    
    def encode(
        self,
        sentences,
        is_query: bool = True,
        instruction_mask=None,
        return_debug_info: bool = False,
        **kwargs
    ):
        """
        编码句子，支持 IGP 处理
        
        参数:
            sentences: 待编码的文本或文本列表
            is_query: 是否为查询（查询会使用 IGP，文档不使用）
            instruction_mask: 指令掩码（可选）
            return_debug_info: 是否返回调试信息（探针注意力等）
            **kwargs: 其他 encode 参数
        
        返回:
            embeddings: token embeddings 列表
            debug_info: 调试信息（如果 return_debug_info=True）
        """
        # 如果不是查询，或者没有 IGP 模块，直接使用 base_model.encode
        if not is_query or (self.probe is None and self.adapter is None and self.gate is None):
            return self.base_model.encode(sentences, is_query=is_query, **kwargs)
        
        # 对于查询，使用 IGP 处理
        # 1. 先使用 base_model 的 tokenizer 获取 input_ids
        # 确保 sentences 是列表
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # 获取 tokenizer
        tokenizer = self.base_model.tokenizer
        
        # 编码文本
        encoded = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.base_model.device)
        attention_mask = encoded['attention_mask'].to(self.base_model.device)
        
        # 如果没有提供 instruction_mask，尝试自动检测
        if instruction_mask is None:
            instruction_mask = self._auto_detect_instruction_mask(input_ids, attention_mask)
        else:
            # 确保 instruction_mask 也在正确的设备上
            if not isinstance(instruction_mask, torch.Tensor):
                instruction_mask = torch.tensor(instruction_mask, device=input_ids.device)
            else:
                instruction_mask = instruction_mask.to(input_ids.device)
        
        # 2. 调用 forward 获取增强后的 embeddings
        with torch.no_grad():
            result = self.forward(
                query_input_ids=input_ids,
                query_attention_mask=attention_mask,
                instruction_mask=instruction_mask,
            )
        
        # 3. 转换为 ColBERT 格式的 embeddings 列表
        token_embeddings = result['token_embeddings']
        
        # 转换为列表格式（每个元素是变长的 tensor）
        embeddings_list = []
        for i in range(token_embeddings.size(0)):
            # 获取有效的 token（非 padding）
            valid_mask = attention_mask[i].bool()
            valid_embeddings = token_embeddings[i][valid_mask]
            embeddings_list.append(valid_embeddings.cpu())
        
        # 4. 收集调试信息
        if return_debug_info:
            debug_info_list = []
            for i in range(token_embeddings.size(0)):
                # 获取 token 文本
                valid_mask = attention_mask[i].bool()
                valid_tokens = input_ids[i][valid_mask]
                token_texts = [tokenizer.decode([tid]) for tid in valid_tokens.cpu().tolist()]
                
                # 获取该样本的注意力分数
                attn_logits = result.get('attn_logits')
                if attn_logits is not None and i < len(attn_logits):
                    sample_attn = attn_logits[i].cpu().numpy()
                else:
                    sample_attn = None
                
                # 收集调试信息（适合端到端训练的字段）
                debug_info = {
                    'token_texts': token_texts,
                    'attn_logits': sample_attn,
                    'debug_stats': result.get('debug_stats', {}),
                }
                debug_info_list.append(debug_info)
            return embeddings_list, debug_info_list
        
        return embeddings_list
    
    def _auto_detect_instruction_mask(self, input_ids, attention_mask):
        """自动检测指令掩码（基于 SEP token）"""
        batch_size, seq_len = input_ids.shape
        instruction_mask = torch.zeros(batch_size, seq_len, dtype=torch.float, device=input_ids.device)
        
        sep_id = self.base_model.tokenizer.sep_token_id
        
        for i in range(batch_size):
            sep_positions = (input_ids[i] == sep_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                first_sep = sep_positions[0].item()
                # SEP 之后的 token 标记为指令
                instruction_mask[i, first_sep + 1:] = 1.0
        
        return instruction_mask
    
    def get_instruction_vector(self, input_ids, attention_mask):
        """获取指令向量（供外部使用）"""
        if self.probe is None:
            return None
        
        query_features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        query_out = self.base_model[0](query_features)
        Q_hidden = query_out['token_embeddings']
        
        probe_output = self.probe(Q_hidden, attention_mask)
        if isinstance(probe_output, tuple):
            inst_vec = probe_output[0]
        else:
            inst_vec = probe_output
        return inst_vec
    
    def tokenize(self, texts, is_query=True, pad=False, task=None, **kwargs):
        """Tokenize 文本，代理到 base_model 的 tokenize 方法"""
        return self.base_model.tokenize(texts, is_query=is_query, pad=pad, task=task, **kwargs)
    
    @property
    def tokenizer(self):
        """获取 tokenizer，代理到 base_model"""
        return self.base_model.tokenizer
    
    @property
    def device(self):
        """获取设备"""
        return next(self.base_model.parameters()).device
    
    def __iter__(self):
        """使模型可迭代，代理到 base_model"""
        return iter(self.base_model)
    
    def __len__(self):
        """返回模型模块数量"""
        return len(self.base_model)
    
    def __getitem__(self, idx):
        """支持索引访问"""
        return self.base_model[idx]
    
    @property
    def model_card_data(self):
        """获取 model_card_data，代理到 base_model"""
        return self.base_model.model_card_data
    
    @model_card_data.setter
    def model_card_data(self, value):
        """设置 model_card_data，代理到 base_model"""
        self.base_model.model_card_data = value
    
    def save_pretrained(self, output_dir: str, **kwargs):
        """保存模型，代理到 base_model"""
        self.base_model.save_pretrained(output_dir, **kwargs)
    
    def save(self, output_dir: str, **kwargs):
        """保存模型（兼容方法），代理到 base_model"""
        self.base_model.save_pretrained(output_dir, **kwargs)
    
    def save_igp_modules(self, save_dir: str):
        """保存 IGP 模块参数"""
        import os
        
        os.makedirs(save_dir, exist_ok=True)
        
        if self.probe is not None:
            probe_path = os.path.join(save_dir, "igp_probe.pt")
            torch.save(self.probe.state_dict(), probe_path)
            
        if self.adapter is not None:
            adapter_path = os.path.join(save_dir, "igp_adapter.pt")
            torch.save(self.adapter.state_dict(), adapter_path)
            
        if self.gate is not None:
            gate_path = os.path.join(save_dir, "igp_gate.pt")
            torch.save(self.gate.state_dict(), gate_path)
    
    def load_igp_modules(self, load_dir: str):
        """加载 IGP 模块参数"""
        import os
        
        if self.probe is not None:
            probe_path = os.path.join(load_dir, "igp_probe.pt")
            if os.path.exists(probe_path):
                self.probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
                
        if self.adapter is not None:
            adapter_path = os.path.join(load_dir, "igp_adapter.pt")
            if os.path.exists(adapter_path):
                self.adapter.load_state_dict(torch.load(adapter_path, map_location='cpu'))
                
        if self.gate is not None:
            gate_path = os.path.join(load_dir, "igp_gate.pt")
            if os.path.exists(gate_path):
                self.gate.load_state_dict(torch.load(gate_path, map_location='cpu'))
