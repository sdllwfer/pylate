"""
IGP Modules - 指令引导探针模块集合

该文件整合了 IGP 架构的所有核心组件，提供统一的接口。

组件:
- InstructionProbe: 指令引导探针模块
- IGPAdapter: IGP 适配器模块
- RatioGate: 门控机制模块

使用示例:
    from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate
    
    probe = InstructionProbe(hidden_size=768)
    adapter = IGPAdapter(hidden_size=768, bottleneck_dim=64)
    gate = RatioGate(hidden_size=768)
"""

from .instruction_probe import InstructionProbe, InstructionProbeConfig
from .igp_adapter import IGPAdapter, IGPAdapterConfig
from .ratio_gate import RatioGate, RatioGateConfig

__all__ = [
    'InstructionProbe',
    'InstructionProbeConfig',
    'IGPAdapter',
    'IGPAdapterConfig', 
    'RatioGate',
    'RatioGateConfig',
]
