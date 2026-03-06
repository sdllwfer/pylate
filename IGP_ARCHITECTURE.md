# ColBERT-IGP 架构设计说明文档

## 1. 概述

ColBERT-IGP (Instruction-Guided Probe) 是对原生 ColBERT 模型的升级，旨在让模型能够从输入序列中提取指令特征，并通过门控机制融合到 Query 向量中，实现"遵从指令的检索"。

## 2. 模块设计

### 2.1 文件结构

```
pylate/models/
├── igp/                          # IGP 模块目录
│   ├── __init__.py               # 模块导出
│   ├── instruction_probe.py       # 指令引导探针
│   ├── igp_adapter.py            # IGP 适配器
│   └── ratio_gate.py             # 门控机制
├── colbert.py                    # 原始 ColBERT (保留)
└── colbert_igp.py               # ColBERT-IGP 模型
```

### 2.2 模块说明

#### 2.2.1 InstructionProbe (instruction_probe.py)

**功能**: 从输入序列中提取指令特征

**结构**:
- 可学习的 `probe_token` 参数 [1, 1, hidden_size]
- TransformerEncoderLayer 作为上下文编码器
- 点积注意力计算 (不使用 Softmax，使用 Sigmoid)

**输入**:
- `query_embeddings`: [batch_size, seq_len, hidden_size]
- `attention_mask`: [batch_size, seq_len]

**输出**:
- `inst_vec`: 指令向量 [batch_size, hidden_size]
- `attn_logits`: 未归一化注意力分数 [batch_size, seq_len]
- `attn_weights`: Sigmoid 注意力权重 [batch_size, seq_len]

#### 2.2.2 IGPAdapter (igp_adapter.py)

**功能**: 在 Transformer 层之间注入指令知识

**结构**:
- 瓶颈结构: hidden_size → bottleneck_dim → hidden_size
- 残差连接和 LayerNorm

**参数**:
- `hidden_size`: 隐藏层维度 (默认 768)
- `bottleneck_dim`: 瓶颈维度 (默认 64)

#### 2.2.3 RatioGate (ratio_gate.py)

**功能**: 控制原始表示和指令增强表示的融合比例

**结构**:
- 可学习参数 `ratio_gate` (初始值 0.0)
- 静态门控: `current_ratio = max_ratio * sigmoid(ratio_gate)`
- 动态门控 (可选): 基于输入预测门控值

**参数**:
- `max_ratio`: 门控最大比率 (默认 0.2，防止指令破坏原语义)
- `use_dynamic`: 是否使用动态门控

## 3. 两阶段训练策略

### 3.1 Phase 1: Probe Warm-up (探针热身)

**目标**: 训练 InstructionProbe 提取指令特征

**冻结策略**:
- 冻结 BERT 主干
- 冻结 Adapter
- 冻结 Gate
- 仅训练 Probe

**训练目标**:
- 仅使用 `aux_loss` (BCE Loss) 进行反向传播
- 忽略排序输出

**早停机制**:
- 每 50 steps 在验证集上评估 `aux_loss`
- 若连续 3 次不下降或 loss < 0.25，立即停止

### 3.2 Phase 2: Joint Training (联合训练)

**目标**: 联合训练所有参数

**解冻策略**:
- 解冻 BERT 主干
- 解冻 Probe
- 解冻 Adapter
- 解冻 Gate

**学习率配置**:
- BERT 主干: 1e-5
- Probe: 1e-5
- Adapter: 1e-5
- Gate: 1e-2 (大学习率，确保门控激活)

**训练目标**:
- 总 Loss = Rank Loss + aux_loss

## 4. 使用方法

### 4.1 训练

```bash
# 使用两阶段训练脚本
python scripts/train_colbert_igp.py \
    --model_name lightonai/GTE-ModernColBERT-v1 \
    --train_data /path/to/train.jsonl \
    --output_dir /path/to/output \
    --phase1_epochs 2 \
    --phase2_epochs 3 \
    --batch_size 16 \
    --base_lr 1e-5 \
    --gate_lr 1e-2
```

### 4.2 参数配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--enable_probe` | 启用 InstructionProbe | True |
| `--enable_adapter` | 启用 IGPAdapter | True |
| `--enable_gate` | 启用 RatioGate | True |
| `--max_ratio` | 门控最大比率 | 0.2 |
| `--bottleneck_dim` | Adapter 瓶颈维度 | 64 |
| `--aux_loss_weight` | 辅助损失权重 | 0.1 |

## 5. 注意事项

1. **门控限制**: 必须限制门控最大值 (默认 0.2)，防止指令破坏原语义
2. **Sigmoid vs Softmax**: 使用 Sigmoid 而非 Softmax，保留多词指令概率
3. **正负样本不平衡**: 使用 `pos_weight=10.0` 处理 BCE Loss 的不平衡
4. **门控学习率**: Phase 2 中为门控分配大学习率 (1e-2)，确保门控能被激活

## 6. 扩展性

IGP 架构支持以下配置:
- 独立启用/禁用各模块 (Probe, Adapter, Gate)
- 静态或动态门控
- 可调的瓶颈维度
- 可调的门控最大比率

所有模块均通过配置文件或命令行参数进行动态配置。

---

## 7. 检查点参数使用指南

### 7.1 参数定义

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--phase1_checkpoint` | Phase 1 检查点路径 | None |
| `--phase2_checkpoint` | Phase 2 检查点路径 | None |

### 7.2 参数说明

- **`--phase1_checkpoint`**: 指定 Phase 1 训练生成的检查点路径。如果为空，则从初始化权重开始训练 Phase 1。
- **`--phase2_checkpoint`**: 指定 Phase 2 训练生成的检查点路径。如果为空，则从当前模型状态开始训练 Phase 2。

### 7.3 配置方法

#### 通过命令行参数

```bash
# 方式1: 只指定 Phase 1 检查点
python scripts/train_colbert_igp.py \
    --phase1_checkpoint /path/to/phase1/checkpoint-5 \
    ...

# 方式2: 同时指定两个阶段的检查点
python scripts/train_colbert_igp.py \
    --phase1_checkpoint /path/to/phase1/checkpoint-10 \
    --phase2_checkpoint /path/to/phase2/checkpoint-3 \
    ...

# 方式3: 只指定 Phase 2 检查点 (Phase 1 从头训练)
python scripts/train_colbert_igp.py \
    --phase2_checkpoint /path/to/phase2/checkpoint-5 \
    ...
```

#### 通过 Shell 脚本配置

编辑 `run_train_colbert_igp.sh` 或 `run_train_colbert_igp_nohup.sh`:

```bash
# Phase 1 检查点路径 (为空则从头训练)
PHASE1_CHECKPOINT=""

# Phase 2 检查点路径 (为空则从头训练)
PHASE2_CHECKPOINT=""

# 示例: 从 Phase 1 检查点恢复继续训练
PHASE1_CHECKPOINT="/home/luwa/Documents/pylate/output/xxx/phase1/checkpoint-5"
```

### 7.4 应用场景

#### 场景1: 完整训练 Phase 1 + Phase 2

```bash
PHASE1_CHECKPOINT=""
PHASE2_CHECKPOINT=""
```

行为:
1. Phase 1 从初始化权重开始训练
2. Phase 2 自动加载 Phase 1 最后的检查点继续训练

#### 场景2: 从 Phase 1 检查点恢复训练

```bash
PHASE1_CHECKPOINT="/path/to/phase1/checkpoint-5"
PHASE2_CHECKPOINT=""
```

行为:
1. 跳过 Phase 1 训练 (从检查点恢复)
2. Phase 2 从该检查点继续训练

#### 场景3: Phase 2 独立训练 (必须指定 Phase 1 检查点)

```bash
ENABLE_PHASE1=false
ENABLE_PHASE2=true
PHASE1_CHECKPOINT="/path/to/phase1/checkpoint-10"
PHASE2_CHECKPOINT=""
```

行为:
1. 验证 Phase 1 检查点是否存在
2. 加载 Phase 1 检查点作为初始模型
3. 执行 Phase 2 训练

> **注意**: Phase 2 独立训练时，必须指定 `--phase1_checkpoint`，否则程序会报错退出。

#### 场景4: 从 Phase 2 检查点恢复训练

```bash
PHASE1_CHECKPOINT=""
PHASE2_CHECKPOINT="/path/to/phase2/checkpoint-3"
```

行为:
1. Phase 1 从初始化权重开始训练
2. Phase 2 从指定检查点继续训练

#### 场景5: 同时恢复两个阶段

```bash
PHASE1_CHECKPOINT="/path/to/phase1/checkpoint-10"
PHASE2_CHECKPOINT="/path/to/phase2/checkpoint-5"
```

行为:
1. 跳过 Phase 1 训练
2. 从 Phase 2 检查点继续训练

### 7.5 注意事项

1. **检查点路径必须完整**: 路径应为检查点目录的绝对路径，例如:
   ```
   /home/luwa/Documents/pylate/output/colbert_igp_train/xxx/phase1/checkpoint-5
   ```

2. **Phase 2 独立训练**: 使用 `--enable_phase1 false --enable_phase2 true` 时，必须同时指定 `--phase1_checkpoint`。

3. **检查点验证**: 程序会自动验证:
   - 检查点路径是否存在
   - 必要文件是否完整 (config.json, model.safetensors)
   - 模型是否可以正常加载

4. **自动查找**: 如果 Phase 2 训练时未指定 `--phase1_checkpoint`，但 Phase 1 已完成训练，程序会自动查找最新的 Phase 1 检查点。

5. **日志输出**: 加载检查点时，会输出详细信息:
   ```
   ============================================================
   📂 正在加载检查点...
      阶段: phase1
      路径: /path/to/checkpoint-5
      名称: checkpoint-5
   ============================================================
      Epoch: 5
   ============================================================
   ✅ 检查点加载成功! [phase1] checkpoint-5
   ============================================================
   ```

### 7.6 错误处理

| 错误情况 | 处理方式 |
|----------|----------|
| 路径不存在 | 抛出 `FileNotFoundError`，显示路径信息 |
| 文件不完整 | 抛出 `FileNotFoundError`，列出缺失文件 |
| 模型加载失败 | 抛出 `RuntimeError`，显示错误详情 |
| Phase 2 独立训练未指定 Phase 1 检查点 | 抛出 `ValueError`，提示必须指定检查点 |

### 7.7 检查点命名规则

检查点按以下规则命名:
```
checkpoint-{epoch}
```

例如:
- `checkpoint-1` - 第 1 个 epoch 完成后
- `checkpoint-5` - 第 5 个 epoch 完成后
- `checkpoint-10` - 第 10 个 epoch 完成后
