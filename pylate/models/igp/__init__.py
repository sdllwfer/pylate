"""
IGP Modules - 指令引导探针模块集合

该文件整合了 IGP 架构的所有核心组件，提供统一的接口。

组件:
- InstructionProbe: 指令引导探针模块
- IGPAdapter: IGP 适配器模块
- RatioGate: 门控机制模块
- V2 版本: 增加参数量的改进版本

使用示例:
    from pylate.models.igp import InstructionProbe, IGPAdapter, RatioGate
    
    probe = InstructionProbe(hidden_size=768)
    adapter = IGPAdapter(hidden_size=768, bottleneck_dim=64)
    gate = RatioGate(hidden_size=768)
"""

from .instruction_probe import InstructionProbe, InstructionProbeConfig
from .igp_adapter import IGPAdapter, IGPAdapterConfig
from .ratio_gate import RatioGate, RatioGateConfig

# V2 版本（增加参数量）
from .instruction_probe_v2 import InstructionProbeV2
from .igp_adapter_v2 import IGPAdapterV2
from .ratio_gate_v2 import RatioGateV2

# V3 版本（动态感知门控，带L1稀疏正则）
from .ratio_gate_v3 import RatioGateV3

# IGP Wrapper
from .igp_wrapper import IGPColBERTWrapper

__all__ = [
    'InstructionProbe',
    'InstructionProbeConfig',
    'IGPAdapter',
    'IGPAdapterConfig', 
    'RatioGate',
    'RatioGateConfig',
    'InstructionProbeV2',
    'IGPAdapterV2',
    'RatioGateV2',
    'RatioGateV3',
    'IGPColBERTWrapper',
]
