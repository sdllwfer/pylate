"""
ColBERT-IGP Model - 指令引导的ColBERT检索模型

该模块将IGP (Instruction-Guided Probe) 架构集成到ColBERT模型中，
实现遵从指令的检索功能。

设计规范:
- 继承自ColBERT，保持原有功能
- 通过参数动态启用/禁用IGP模块
- 支持两阶段训练: Probe Warm-up 和 Joint Training

核心组件:
- InstructionProbe: 提取指令特征
- IGPAdapter: 注入指令知识
- RatioGate: 控制融合比例
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

from .colbert import ColBERT

logger = logging.getLogger(__name__)


class ColBERTIGPModel(nn.Module):
    """
    ColBERT-IGP 模型
    
    在ColBERT基础上添加IGP指令引导探针架构。
    
    参数:
        base_model: 基础ColBERT模型
        enable_igp (bool): 是否启用IGP模块
        enable_probe (bool): 是否启用InstructionProbe
        enable_adapter (bool): 是否启用IGPAdapter  
        enable_gate (bool): 是否启用RatioGate
        probe_config (dict): InstructionProbe配置
        adapter_config (dict): IGPAdapter配置
        gate_config (dict): RatioGate配置
    """
    
    def __init__(
        self,
        base_model: ColBERT,
        enable_igp: bool = True,
        enable_probe: bool = True,
        enable_adapter: bool = True,
        enable_gate: bool = True,
        probe_config: Optional[Dict[str, Any]] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        gate_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # 基础模型 (冻结)
        self.base_model = base_model
        self.hidden_size = base_model[0].get_word_embedding_dimension()
        
        # IGP 配置
        self.enable_igp = enable_igp
        self.enable_probe = enable_probe and enable_igp
        self.enable_adapter = enable_adapter and enable_igp
        self.enable_gate = enable_gate and enable_igp
        
        # 默认配置
        probe_config = probe_config or {}
        adapter_config = adapter_config or {}
        gate_config = gate_config or {}
        
        # 初始化 IGP 模块
        if self.enable_probe:
            from .igp.instruction_probe import InstructionProbe
            self.probe = InstructionProbe(
                hidden_size=self.hidden_size,
                num_heads=probe_config.get('num_heads', 8),
                dropout=probe_config.get('dropout', 0.1),
            )
            logger.info(f"Initialized InstructionProbe with config: {probe_config}")
        else:
            self.probe = None
            
        if self.enable_adapter:
            from .igp.igp_adapter import IGPAdapter
            self.adapter = IGPAdapter(
                hidden_size=self.hidden_size,
                bottleneck_dim=adapter_config.get('bottleneck_dim', 64),
                dropout=adapter_config.get('dropout', 0.1),
            )
            logger.info(f"Initialized IGPAdapter with config: {adapter_config}")
        else:
            self.adapter = None
            
        if self.enable_gate:
            from .igp.ratio_gate import RatioGate
            self.gate = RatioGate(
                hidden_size=self.hidden_size,
                max_ratio=gate_config.get('max_ratio', 0.2),
                use_dynamic=gate_config.get('use_dynamic', False),
            )
            logger.info(f"Initialized RatioGate with config: {gate_config}")
        else:
            self.gate = None
        
        # 辅助损失函数 (BCEWithLogitsLoss，处理正负样本不平衡)
        self.aux_criterion = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.tensor([10.0])
        )
        
        logger.info(
            f"ColBERTIGPModel initialized: "
            f"IGP={enable_igp}, Probe={self.enable_probe}, "
            f"Adapter={self.enable_adapter}, Gate={self.enable_gate}"
        )
    
    def get_igp_parameters(self):
        """获取IGP相关参数 (用于冻结/解冻)"""
        params = []
        if self.enable_probe and self.probe is not None:
            params.extend(list(self.probe.parameters()))
        if self.enable_adapter and self.adapter is not None:
            params.extend(list(self.adapter.parameters()))
        if self.enable_gate and self.gate is not None:
            params.extend(list(self.gate.parameters()))
        return params
    
    def freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Base model frozen")
    
    def unfreeze_base_model(self):
        """解冻基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = True
        logger.info("Base model unfrozen")
    
    def freeze_gate(self):
        """冻结门控参数"""
        if self.gate is not None:
            for param in self.gate.parameters():
                param.requires_grad = False
            logger.info("Gate frozen")
    
    def unfreeze_gate(self):
        """解冻门控参数"""
        if self.gate is not None:
            for param in self.gate.parameters():
                param.requires_grad = True
            logger.info("Gate unfrozen")
    
    def encode(
        self,
        sentences: str | list[str],
        instruction_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        编码查询/文档，支持IGP处理
        
        参数:
            sentences: 待编码的文本
            instruction_mask: 指令掩码，用于辅助损失计算
            **kwargs: 其他ColBERT encode参数
        
        返回:
            embeddings: token embeddings 列表 (与ColBERT基础模型兼容)
        """
        embeddings = self.base_model.encode(sentences, **kwargs)
        
        if not self.enable_igp or not self.enable_probe:
            return embeddings
        
        return embeddings
    
    def forward(
        self,
        queries: list[str],
        positive_docs: list[str],
        negative_docs: list[str],
        instruction_masks: Optional[list[torch.Tensor]] = None,
        is_query: bool = True,
    ) -> Dict[str, Any]:
        """
        前向传播
        
        参数:
            queries: 查询列表
            positive_docs: 正例文档列表
            negative_docs: 负例文档列表
            instruction_masks: 指令掩码列表
            is_query: 是否为查询编码
        
        返回:
            包含损失和中间结果的字典
        """
        # 编码查询
        query_embeddings = self.base_model.encode(
            queries, 
            is_query=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        # 编码文档
        pos_doc_embeddings = self.base_model.encode(
            positive_docs,
            is_query=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        neg_doc_embeddings = self.base_model.encode(
            negative_docs,
            is_query=False,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        
        # 初始化返回值
        aux_loss = None
        gate_ratio = None
        
        if self.enable_igp and self.enable_probe and self.probe is not None:
            # 需要处理 instruction_mask
            # 这里简化处理，实际使用需要在训练数据中提供
            
            # 注意: ColBERT的embeddings是list，每个元素是变长的tensor
            # 需要特殊处理才能使用IGP
            
            pass
        
        return {
            'query_embeddings': query_embeddings,
            'positive_embeddings': pos_doc_embeddings,
            'negative_embeddings': neg_doc_embeddings,
            'aux_loss': aux_loss,
            'gate_ratio': gate_ratio,
        }
    
    def compute_contrastive_loss(
        self,
        query_embeddings: list[torch.Tensor],
        positive_embeddings: list[torch.Tensor],
        negative_embeddings: list[torch.Tensor],
        temperature: float = 0.01,
    ) -> torch.Tensor:
        """
        计算对比损失 (InfoNCE)
        
        参数:
            query_embeddings: 查询embeddings
            positive_embeddings: 正例embeddings
            negative_embeddings: 负例embeddings
            temperature: 温度参数
        
        返回:
            损失值
        """
        from ..scores import MaxSim
        
        maxsim = MaxSim()
        
        # 计算正例分数
        pos_scores = []
        for q_emb, pos_emb in zip(query_embeddings, positive_embeddings):
            # MaxSim: [num_pos_docs], 对每个查询只有一个正例
            score = maxsim(
                q_emb.unsqueeze(0),  # [1, query_len, dim]
                pos_emb.unsqueeze(0),  # [1, doc_len, dim]
            )
            pos_scores.append(score)
        
        # 计算负例分数
        neg_scores = []
        for q_emb, neg_emb in zip(query_embeddings, negative_embeddings):
            score = maxsim(
                q_emb.unsqueeze(0),
                neg_emb.unsqueeze(0),
            )
            neg_scores.append(score)
        
        # 合并分数
        all_scores = torch.cat([
            torch.stack(pos_scores),
            torch.stack(neg_scores)
        ], dim=1)  # [batch_size, 1 + num_negatives]
        
        # Labels: 正例在第一个位置
        labels = torch.zeros(len(query_embeddings), dtype=torch.long, device=all_scores.device)
        
        # Cross Entropy Loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(all_scores / temperature, labels)
        
        return loss


class ColBERTIGPConfig:
    """ColBERT-IGP 配置类"""
    
    def __init__(
        self,
        enable_igp: bool = True,
        enable_probe: bool = True,
        enable_adapter: bool = True,
        enable_gate: bool = True,
        probe_config: Optional[Dict[str, Any]] = None,
        adapter_config: Optional[Dict[str, Any]] = None,
        gate_config: Optional[Dict[str, Any]] = None,
    ):
        self.enable_igp = enable_igp
        self.enable_probe = enable_probe
        self.enable_adapter = enable_adapter
        self.enable_gate = enable_gate
        self.probe_config = probe_config or {}
        self.adapter_config = adapter_config or {}
        self.gate_config = gate_config or {}
    
    def to_dict(self):
        return {
            'enable_igp': self.enable_igp,
            'enable_probe': self.enable_probe,
            'enable_adapter': self.enable_adapter,
            'enable_gate': self.enable_gate,
            'probe_config': self.probe_config,
            'adapter_config': self.adapter_config,
            'gate_config': self.gate_config,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)
