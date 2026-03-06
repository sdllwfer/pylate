"""
ColBERT-IGP 两阶段训练脚本

实现 "Probe Warm-up" + "Joint Training" 两阶段训练策略。

Phase 1: 探针热身
- 冻结除 probe 和 instruction_attention 外的所有参数
- 仅使用 aux_loss 进行反向传播
- 早停机制: 每50步评估，若连续3次不下降或 loss < 0.25 则停止

Phase 2: 联合训练
- 解冻所有参数
- 为门控分配独立学习率 (较大)
- 总 Loss = Rank Loss + aux_loss
"""

import os

# =========================================================
# ⚠️ 1. 网络与环境配置
# =========================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# =========================================================

import sys

conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env != 'pylate':
    import subprocess
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    conda_path = subprocess.run(['which', 'conda'], capture_output=True, text=True).stdout.strip()
    if conda_path:
        env_python = f"{os.path.dirname(conda_path)}/envs/pylate/bin/python"
        if os.path.exists(env_python):
            os.execv(env_python, [sys.executable] + sys.argv)

import argparse
import json
import sys

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

from datetime import datetime
from typing import Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"⚠️ matplotlib 不可用: {e}")

import torch
import torch.nn as nn
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

from pylate import losses, models, utils
from pylate.models.igp import (
    InstructionProbe,
    IGPAdapter,
    RatioGate,
)


class EarlyStoppingCallback(TrainerCallback):
    """
    Phase 1 早停回调
    
    - 每 eval_steps 评估一次
    - 连续 patience 次不下降则停止
    - 或者 loss < threshold 则停止
    """
    
    def __init__(
        self,
        eval_steps: int = 50,
        patience: int = 3,
        threshold: float = 0.25,
    ):
        self.eval_steps = eval_steps
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float('inf')
        self.no_improve_count = 0
        self.should_stop = False
        self.best_epoch = 0
        self.last_valid_loss = None
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        current_loss = metrics.get('eval_loss', float('inf'))
        epoch = int(state.global_step / self.eval_steps) if state.global_step else 0
        self.last_valid_loss = current_loss
        
        print(f"\n📊 Phase 1 评估: eval_loss = {current_loss:.4f}, best = {self.best_loss:.4f}")
        
        if current_loss < self.threshold:
            print(f"✅ 早停触发: eval_loss ({current_loss:.4f}) < threshold ({self.threshold})")
            print(f"   最佳epoch: {self.best_epoch}, 最佳验证损失: {self.best_loss:.6f}")
            if self.last_valid_loss is not None and self.last_valid_loss != float('inf'):
                print(f"   最后有效epoch: {epoch}, 验证损失: {self.last_valid_loss:.6f}")
            control.should_training_stop = True
            self.should_stop = True
            return
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.no_improve_count = 0
            print(f"📈 损失改善: {self.best_loss:.4f}")
        else:
            self.no_improve_count += 1
            print(f"📉 损失未改善: {self.no_improve_count}/{self.patience}")
            
            if self.no_improve_count >= self.patience:
                print(f"\n✅ 早停触发: 连续 {self.patience} 次未改善")
                print(f"   最佳epoch: {self.best_epoch}, 最佳验证损失: {self.best_loss:.6f}")
                if self.last_valid_loss is not None and self.last_valid_loss != float('inf'):
                    print(f"   最后有效epoch: {epoch}, 验证损失: {self.last_valid_loss:.6f}")
                control.should_training_stop = True
                self.should_stop = True


class Phase2EarlyStoppingCallback(TrainerCallback):
    """
    Phase 2 早停回调
    
    监控验证集性能指标，当连续多个epoch没有改善时自动停止训练
    
    参数:
        patience: 连续多少个epoch没有改善则停止
        threshold: 改善的最小阈值
        mode: 'min' 表示监控最小值(如loss), 'max' 表示监控最大值(如accuracy)
    """
    
    def __init__(
        self,
        patience: int = 3,
        threshold: float = 0.001,
        threshold_mode: str = 'rel',
        mode: str = 'min',
    ):
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.mode = mode
        
        if mode == 'min':
            self.best_metric = float('inf')
            self.is_better = self._is_better_min
        else:
            self.best_metric = float('-inf')
            self.is_better = self._is_better_max
        
        self.no_improve_count = 0
        self.should_stop = False
        self.best_epoch = 0
        
        self.eval_history = []
        self.last_valid_metric = None
    
    def _is_better_min(self, current, best):
        if self.threshold_mode == 'rel':
            return current < best - best * self.threshold
        else:
            return current < best - self.threshold
    
    def _is_better_max(self, current, best):
        if self.threshold_mode == 'rel':
            return current > best + best * self.threshold
        else:
            return current > best + self.threshold
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        current_metric = metrics.get('eval_loss', float('inf'))
        epoch = int(state.epoch) if state.epoch else 0
        
        self.eval_history.append({'epoch': epoch, 'metric': current_metric})
        self.last_valid_metric = current_metric
        
        print(f"\n📊 Phase 2 评估: epoch={epoch}, eval_loss={current_metric:.6f}, best={self.best_metric:.6f}")
        
        if self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.no_improve_count = 0
            print(f"📈 性能改善! best_epoch={self.best_epoch}, best_metric={self.best_metric:.6f}")
        else:
            self.no_improve_count += 1
            print(f"📉 性能未改善: {self.no_improve_count}/{self.patience}")
            
            if self.no_improve_count >= self.patience:
                print(f"\n🛑 早停触发! 连续 {self.patience} 个epoch性能未改善")
                print(f"   最佳epoch: {self.best_epoch}, 最佳验证损失: {self.best_metric:.6f}")
                if self.last_valid_metric is not None and self.last_valid_metric != float('inf'):
                    print(f"   最后有效epoch: {epoch}, 验证损失: {self.last_valid_metric:.6f}")
                control.should_training_stop = True
                self.should_stop = True


class IGPTrainingCallbacks:
    """IGP训练回调集合"""
    
    @staticmethod
    def create_phase1_callbacks(output_dir, eval_dataset):
        """创建Phase 1 (Probe Warm-up) 的回调"""
        callbacks = []
        return callbacks
    
    @staticmethod
    def create_phase2_callbacks(output_dir):
        """创建Phase 2 (Joint Training) 的回调"""
        callbacks = []
        return callbacks


class LossTracker(TrainerCallback):
    """损失追踪回调 - 记录并绘制训练/验证损失曲线"""
    
    def __init__(self, output_dir, trainer, phase_name="training"):
        self.output_dir = output_dir
        self.trainer = trainer
        self.phase_name = phase_name
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.batch_losses = []
        self.current_step = 0
        self.epochs = []
        self.current_train_loss = None
        self.history_file = os.path.join(output_dir, "loss_history.json")
        
        os.makedirs(output_dir, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.current_train_loss = logs.get('loss')
                if 'step' in logs:
                    step = logs.get('step', self.current_step)
                    self.batch_losses.append({
                        'step': step,
                        'train_loss': self.current_train_loss,
                        'phase': self.phase_name
                    })
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = 0
        self.batch_losses = []
    
    def on_step_end(self, args, state, control, outputs=None, **kwargs):
        if outputs is not None and hasattr(outputs, 'loss'):
            loss = outputs.loss.item() if hasattr(outputs.loss, 'item') else outputs.loss
            self.current_step = state.global_step
            self.batch_losses.append({
                'step': self.current_step,
                'train_loss': loss,
                'phase': self.phase_name
            })
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics.get('eval_loss')
            epoch = int(state.epoch) if state.epoch else 0
            
            eval_step = state.global_step
            
            while len(self.eval_losses) < epoch:
                self.eval_losses.append(None)
                self.eval_steps.append(None)
            
            self.eval_losses.append(eval_loss)
            self.eval_steps.append(eval_step)
            
            self._save_history()
            self._plot_losses()
            print(f"\n📊 验证损失已记录: step={eval_step}, epoch={epoch}, eval_loss={eval_loss:.6f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else (len(self.epochs) + 1)
        
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            if self.current_train_loss is not None:
                self.train_losses.append(self.current_train_loss)
                self.train_steps.append(state.global_step)
            self.current_train_loss = None
        
        self._save_history()
        self._plot_losses()
    
    def _save_history(self):
        history = {
            'phase': self.phase_name,
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'train_steps': self.train_steps,
            'eval_losses': self.eval_losses,
            'eval_steps': self.eval_steps,
            'batch_losses': self.batch_losses
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _save_csv(self):
        step_csv_path = os.path.join(self.output_dir, "loss_history_step.csv")
        
        with open(step_csv_path, 'w') as f:
            f.write("step,phase,train_loss\n")
            for batch in self.batch_losses:
                f.write(f"{batch['step']},{batch.get('phase', '')},{batch['train_loss']}\n")
        
        epoch_csv_path = os.path.join(self.output_dir, "loss_history_epoch.csv")
        
        eval_losses_filtered = [l for l in self.eval_losses if l is not None]
        
        with open(epoch_csv_path, 'w') as f:
            f.write("epoch,train_loss,eval_loss\n")
            for i in range(len(self.train_losses)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) else ""
                eval_loss = eval_losses_filtered[i] if i < len(eval_losses_filtered) else ""
                f.write(f"{i+1},{train_loss},{eval_loss}\n")
        
        print(f"📊 Step级损失数据已保存: {step_csv_path}")
        print(f"📊 Epoch级损失数据已保存: {epoch_csv_path}")
    
    def _plot_losses(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        batch_steps = [b['step'] for b in self.batch_losses]
        batch_losses = [b['train_loss'] for b in self.batch_losses]
        
        eval_steps = [s for s in self.eval_steps if s is not None]
        eval_losses_filtered = [l for l in self.eval_losses if l is not None]
        
        if not batch_losses and not eval_losses_filtered:
            return
        
        self._plot_step_curve(batch_steps, batch_losses, eval_steps, eval_losses_filtered)
        
        self._plot_epoch_curve()
        
        self._save_csv()
    
    def _plot_step_curve(self, batch_steps, batch_losses, eval_steps, eval_losses_filtered):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        if batch_losses:
            ax.plot(batch_steps, batch_losses, 'b-', alpha=0.3, label='Training Loss (per step)', linewidth=0.8)
            if len(batch_steps) > 10:
                window = min(50, len(batch_losses) // 10)
                if window > 1:
                    smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
                    smoothed_steps = batch_steps[window-1:]
                    ax.plot(smoothed_steps, smoothed, 'b-', label=f'Smoothed (window={window})', linewidth=2)
        
        if eval_steps and eval_losses_filtered:
            ax.plot(eval_steps, eval_losses_filtered, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Loss per Step - {self.phase_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, "loss_curve_step.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Step级损失曲线已保存: {chart_path}")
    
    def _plot_epoch_curve(self):
        if not self.train_losses:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = list(range(1, len(self.train_losses) + 1))
        eval_losses_filtered = [l for l in self.eval_losses if l is not None]
        
        ax.plot(epochs, self.train_losses, 'b-o', 
                label='Training Loss', linewidth=2, markersize=8, markerfacecolor='blue')
        
        if len(eval_losses_filtered) == len(self.train_losses):
            ax.plot(epochs, eval_losses_filtered, 'r-s', 
                    label='Validation Loss', linewidth=2, markersize=8, markerfacecolor='red')
            
            if len(self.train_losses) > 1:
                train_gap = self.train_losses[0] - self.train_losses[-1]
                eval_gap = eval_losses_filtered[0] - eval_losses_filtered[-1]
                if train_gap > 0 and eval_gap > 0:
                    ax.text(0.02, 0.02, '✓ Both losses decreasing', 
                            transform=ax.transAxes, fontsize=10, 
                            color='green', fontweight='bold',
                            verticalalignment='bottom')
                elif train_gap > 0 and eval_gap < 0:
                    ax.text(0.02, 0.98, '⚠️ Possible Overfitting', 
                            transform=ax.transAxes, fontsize=11, 
                            color='orange', fontweight='bold',
                            verticalalignment='top')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training vs Validation Loss per Epoch - {self.phase_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        ax.set_xlim(left=0.5, right=len(epochs) + 0.5)
        
        plt.tight_layout()
        
        chart_path = os.path.join(self.output_dir, "loss_curve_epoch.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Epoch级损失曲线已保存: {chart_path}")


class BestModelCallback(TrainerCallback):
    """自定义回调：保存验证集损失最低的模型"""
    
    def __init__(self, output_dir, model, phase_name="training", igp_modules=None):
        self.output_dir = output_dir
        self.phase_name = phase_name
        self.model = model
        self.igp_modules = igp_modules
        self.best_eval_loss = float('inf')
        self.best_model_path = None
        self.best_epoch = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(output_dir, f"best_model_{phase_name}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics.get('eval_loss')
            
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.best_epoch = int(state.epoch) if state.epoch else 0
                
                import shutil
                if os.path.exists(self.save_dir):
                    shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)
                
                self.model.save(self.save_dir)
                
                if self.igp_modules is not None:
                    self._save_igp_modules(self.save_dir)
                
                print(f"\n🏆 保存最佳模型 ({self.phase_name}, epoch={self.best_epoch}, val_loss: {eval_loss:.6f})")
                print(f"   保存路径: {self.save_dir}")
    
    def _save_igp_modules(self, save_dir):
        """保存 IGP 模块参数"""
        if self.igp_modules is None:
            return
        
        probe, adapter, gate = self.igp_modules
        
        igp_state = {}
        
        if probe is not None and hasattr(probe, 'state_dict'):
            probe_path = os.path.join(save_dir, "igp_probe.pt")
            torch.save(probe.state_dict(), probe_path)
            igp_state['probe'] = 'igp_probe.pt'
            print(f"   ✅ Probe 参数已保存: {probe_path}")
        
        if adapter is not None and hasattr(adapter, 'state_dict'):
            adapter_path = os.path.join(save_dir, "igp_adapter.pt")
            torch.save(adapter.state_dict(), adapter_path)
            igp_state['adapter'] = 'igp_adapter.pt'
            print(f"   ✅ Adapter 参数已保存: {adapter_path}")
        
        if gate is not None and hasattr(gate, 'state_dict'):
            gate_path = os.path.join(save_dir, "igp_gate.pt")
            torch.save(gate.state_dict(), gate_path)
            igp_state['gate'] = 'igp_gate.pt'
            print(f"   ✅ Gate 参数已保存: {gate_path}")
        
        if igp_state:
            igp_info = {
                'phase': self.phase_name,
                'best_epoch': self.best_epoch,
                'best_eval_loss': self.best_eval_loss,
                'modules': igp_state,
                'config': {
                    'enable_probe': probe is not None,
                    'enable_adapter': adapter is not None,
                    'enable_gate': gate is not None,
                }
            }
            igp_info_path = os.path.join(save_dir, "igp_info.json")
            with open(igp_info_path, 'w') as f:
                json.dump(igp_info, f, indent=2)
            print(f"   ✅ IGP 配置已保存: {igp_info_path}")


class EpochCheckpointCallback(TrainerCallback):
    """自定义回调：每个epoch完成后保存checkpoint，按epoch编号命名"""
    
    def __init__(self, output_dir, model, phase_name="training", igp_modules=None):
        self.output_dir = output_dir
        self.model = model
        self.phase_name = phase_name
        self.igp_modules = igp_modules
        self.saved_checkpoints = []
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        if epoch > 0:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{epoch}")
            self.model.save(checkpoint_dir)
            
            if self.igp_modules is not None:
                self._save_igp_modules(checkpoint_dir, epoch)
            
            self.saved_checkpoints.append(epoch)
            print(f"💾 保存 checkpoint: {checkpoint_dir} (epoch {epoch})")
    
    def _save_igp_modules(self, checkpoint_dir, epoch):
        """保存 IGP 模块参数"""
        if self.igp_modules is None:
            return
        
        probe, adapter, gate = self.igp_modules
        
        igp_state = {}
        
        if probe is not None and hasattr(probe, 'state_dict'):
            probe_path = os.path.join(checkpoint_dir, "igp_probe.pt")
            torch.save(probe.state_dict(), probe_path)
            igp_state['probe'] = 'igp_probe.pt'
        
        if adapter is not None and hasattr(adapter, 'state_dict'):
            adapter_path = os.path.join(checkpoint_dir, "igp_adapter.pt")
            torch.save(adapter.state_dict(), adapter_path)
            igp_state['adapter'] = 'igp_adapter.pt'
        
        if gate is not None and hasattr(gate, 'state_dict'):
            gate_path = os.path.join(checkpoint_dir, "igp_gate.pt")
            torch.save(gate.state_dict(), gate_path)
            igp_state['gate'] = 'igp_gate.pt'
        
        if igp_state:
            igp_info = {
                'phase': self.phase_name,
                'epoch': epoch,
                'modules': igp_state,
                'config': {
                    'enable_probe': probe is not None,
                    'enable_adapter': adapter is not None,
                    'enable_gate': gate is not None,
                }
            }
            igp_info_path = os.path.join(checkpoint_dir, "igp_info.json")
            with open(igp_info_path, 'w') as f:
                json.dump(igp_info, f, indent=2)


class IGPLoss(nn.Module):
    """
    IGP 组合损失
    
    结合对比损失和辅助损失 (BCE Loss for instruction detection)
    注意: 此类作为 SentenceTransformerTrainer 的 loss 接口，
    需要返回单个标量损失值。
    """
    
    def __init__(
        self,
        base_loss,
        probe: Optional[InstructionProbe] = None,
        adapter: Optional[IGPAdapter] = None,
        gate: Optional[RatioGate] = None,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        self.aux_loss_weight = aux_loss_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss(
            reduction='none',
            pos_weight=torch.tensor([10.0])
        )
    
    def forward(self, sentence_features: list, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        base_loss = self.base_loss(sentence_features, labels)
        return base_loss


class IGPTrainer:
    """
    IGP 两阶段训练器
    
    Phase 1: Probe Warm-up - 仅训练 probe 参数
    Phase 2: Joint Training - 训练所有参数
    """
    
    def __init__(
        self,
        model_name: str,
        train_data_path: str,
        output_dir: str,
        device: str = 'cuda:0',
        phase1_epochs: int = 2,
        phase2_epochs: int = 3,
        batch_size: int = 16,
        eval_ratio: float = 0.1,
        base_lr: float = 1e-5,
        gate_lr: float = 1e-2,
        eval_steps: int = 50,
        enable_phase1: bool = True,
        enable_phase2: bool = True,
        enable_probe: bool = True,
        enable_adapter: bool = True,
        enable_gate: bool = True,
        max_ratio: float = 0.2,
        bottleneck_dim: int = 64,
        aux_loss_weight: float = 0.1,
        phase2_patience: int = 3,
        phase2_early_stop_threshold: float = 0.001,
        save_total_limit: int = 3,
        phase1_checkpoint: str = None,
        phase2_checkpoint: str = None,
    ):
        self.model_name = model_name
        self.train_data_path = train_data_path
        self.output_dir = output_dir
        self.device = device
        self.phase1_epochs = phase1_epochs
        self.phase2_epochs = phase2_epochs
        self.batch_size = batch_size
        self.eval_ratio = eval_ratio
        self.base_lr = base_lr
        self.gate_lr = gate_lr
        self.eval_steps = eval_steps
        self.enable_phase1 = enable_phase1
        self.enable_phase2 = enable_phase2
        self.save_total_limit = save_total_limit
        self.enable_probe = enable_probe
        self.enable_adapter = enable_adapter
        self.enable_gate = enable_gate
        self.max_ratio = max_ratio
        self.bottleneck_dim = bottleneck_dim
        self.aux_loss_weight = aux_loss_weight
        self.phase2_patience = phase2_patience
        self.phase2_early_stop_threshold = phase2_early_stop_threshold
        self.phase1_checkpoint = phase1_checkpoint
        self.phase2_checkpoint = phase2_checkpoint
        
        # 加载数据和模型
        self.train_dataset = None
        self.eval_dataset = None
        self.base_model = None
        self.probe = None
        self.adapter = None
        self.gate = None
        
    def load_data(self):
        """加载训练和验证数据"""
        print(f"📂 加载训练数据: {self.train_data_path}")
        
        data_list = []
        with open(self.train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = eval(line.strip())
                
                query = item.get('query', '')
                positives = item.get('pos', [])
                negatives = item.get('neg', [])
                
                if positives and negatives:
                    for pos_doc in positives:
                        for neg_doc in negatives:
                            data_list.append({
                                'query': query,
                                'document': pos_doc,
                                'negative': neg_doc,
                            })
        
        dataset = Dataset.from_list(data_list)
        splits = dataset.train_test_split(test_size=self.eval_ratio)
        self.train_dataset = splits['train']
        self.eval_dataset = splits['test']
        
        print(f"✅ 训练集: {len(self.train_dataset)} 样本")
        print(f"✅ 验证集: {len(self.eval_dataset)} 样本")
    
    def load_model(self):
        """加载基础模型"""
        print(f"📥 加载基础模型: {self.model_name}")
        self.base_model = models.ColBERT(model_name_or_path=self.model_name)
        
        hidden_size = self.base_model[0].get_word_embedding_dimension()
        
        # 初始化 IGP 模块
        if self.enable_probe:
            self.probe = InstructionProbe(
                hidden_size=hidden_size,
                num_heads=8,
                dropout=0.1,
            )
            print(f"✅ InstructionProbe 初始化完成 (hidden_size={hidden_size})")
        
        if self.enable_adapter:
            self.adapter = IGPAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=self.bottleneck_dim,
                dropout=0.1,
            )
            print(f"✅ IGPAdapter 初始化完成 (bottleneck_dim={self.bottleneck_dim})")
        
        if self.enable_gate:
            self.gate = RatioGate(
                hidden_size=hidden_size,
                max_ratio=self.max_ratio,
                use_dynamic=False,
            )
            print(f"✅ RatioGate 初始化完成 (max_ratio={self.max_ratio})")
    
    def load_checkpoint(self, checkpoint_path: str, phase: str = "phase1"):
        """
        加载检查点文件
        
        Args:
            checkpoint_path: 检查点目录路径
            phase: 检查点所属阶段 ("phase1" 或 "phase2")
            
        Returns:
            加载的 epoch 数
        """
        import json
        import torch
        
        checkpoint_name = os.path.basename(checkpoint_path)
        
        print(f"\n{'=' * 60}")
        print(f"📂 正在加载检查点...")
        print(f"   阶段: {phase}")
        print(f"   路径: {checkpoint_path}")
        print(f"   名称: {checkpoint_name}")
        print(f"{'=' * 60}")
        
        # 验证路径是否存在
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点加载失败: 路径不存在")
            print(f"   路径: {checkpoint_path}")
            raise FileNotFoundError(f"❌ 检查点路径不存在: {checkpoint_path}")
        
        # 验证必要的文件
        required_files = ['config.json', 'model.safetensors']
        missing_files = []
        for f in required_files:
            fpath = os.path.join(checkpoint_path, f)
            if not os.path.exists(fpath):
                missing_files.append(f)
        
        if missing_files:
            print(f"❌ 检查点加载失败: 文件不完整")
            print(f"   缺少文件: {missing_files}")
            raise FileNotFoundError(f"❌ 检查点文件不完整，缺少: {missing_files}")
        
        # 加载模型
        try:
            self.base_model = models.ColBERT(checkpoint_path)
            print(f"✅ 检查点模型已加载")
        except Exception as e:
            print(f"❌ 检查点加载失败: 模型加载错误")
            print(f"   错误: {str(e)}")
            raise RuntimeError(f"❌ 加载模型失败: {str(e)}")
        
        # 提取 epoch 信息
        loaded_epoch = None
        config_path = os.path.join(checkpoint_path, 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    loaded_epoch = config.get('epoch', None)
                    if loaded_epoch:
                        print(f"   Epoch: {loaded_epoch}")
            except:
                pass
        
        # 加载 IGP 模块状态
        igp_info_path = os.path.join(checkpoint_path, 'igp_info.json')
        if os.path.exists(igp_info_path):
            try:
                with open(igp_info_path, 'r') as f:
                    igp_info = json.load(f)
                    print(f"   IGP阶段: {igp_info.get('phase', 'unknown')}")
                    
                    # 初始化 IGP 模块并加载参数
                    self._load_igp_modules(checkpoint_path, igp_info)
            except Exception as e:
                print(f"   ⚠️ IGP模块加载跳过: {str(e)}")
        
        print(f"{'=' * 60}")
        print(f"✅ 检查点加载成功! [{phase}] {checkpoint_name}")
        print(f"{'=' * 60}")
        return loaded_epoch
    
    def _load_igp_modules(self, checkpoint_path, igp_info):
        """从检查点加载 IGP 模块参数"""
        hidden_size = self.base_model[0].get_word_embedding_dimension()
        
        modules = igp_info.get('modules', {})
        
        if 'probe' in modules and self.enable_probe:
            probe_path = os.path.join(checkpoint_path, modules['probe'])
            if os.path.exists(probe_path):
                if self.probe is None:
                    from pylate.models.igp import InstructionProbe
                    self.probe = InstructionProbe(
                        hidden_size=hidden_size,
                        num_heads=8,
                        dropout=0.1,
                    )
                self.probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
                print(f"   ✅ Probe 参数已加载")
        
        if 'adapter' in modules and self.enable_adapter:
            adapter_path = os.path.join(checkpoint_path, modules['adapter'])
            if os.path.exists(adapter_path):
                if self.adapter is None:
                    from pylate.models.igp import IGPAdapter
                    self.adapter = IGPAdapter(
                        hidden_size=hidden_size,
                        bottleneck_dim=self.bottleneck_dim,
                        dropout=0.1,
                    )
                self.adapter.load_state_dict(torch.load(adapter_path, map_location='cpu'))
                print(f"   ✅ Adapter 参数已加载")
        
        if 'gate' in modules and self.enable_gate:
            gate_path = os.path.join(checkpoint_path, modules['gate'])
            if os.path.exists(gate_path):
                if self.gate is None:
                    from pylate.models.igp import RatioGate
                    self.gate = RatioGate(
                        hidden_size=hidden_size,
                        max_ratio=self.max_ratio,
                        use_dynamic=False,
                    )
                self.gate.load_state_dict(torch.load(gate_path, map_location='cpu'))
                print(f"   ✅ Gate 参数已加载")
    
    def should_skip_phase1(self):
        """检查是否需要跳过 Phase 1"""
        if self.phase1_checkpoint is None:
            return False
        return True
    
    def should_skip_phase2(self):
        """检查是否需要跳过 Phase 2"""
        if self.phase2_checkpoint is None:
            return False
        return True
    
    def freeze_all_except_probe(self):
        """冻结除 probe 外的所有参数"""
        # 冻结基础模型
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 冻结 adapter
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = False
        
        # 冻结 gate
        if self.gate is not None:
            for param in self.gate.parameters():
                param.requires_grad = False
        
        # 确保 probe 可训练
        if self.probe is not None:
            for param in self.probe.parameters():
                param.requires_grad = True
        
        print("🔒 Phase 1: 已冻结除 probe 外的所有参数")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        # 解冻基础模型
        for param in self.base_model.parameters():
            param.requires_grad = True
        
        # 解冻 adapter
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = True
        
        # 解冻 gate
        if self.gate is not None:
            for param in self.gate.parameters():
                param.requires_grad = True
        
        # 解冻 probe
        if self.probe is not None:
            for param in self.probe.parameters():
                param.requires_grad = True
        
        print("🔓 Phase 2: 已解冻所有参数")
    
    def train_phase1(self):
        """
        Phase 1: Probe Warm-up
        
        使用对比损失训练，但给 probe 设置更大的学习率
        """
        print("\n" + "=" * 60)
        print("🚀 Phase 1: 探针热身 (Probe Warm-up)")
        print("=" * 60)
        
        # Phase 1: 不冻结基础模型，但给基础模型极小学习率，给 probe 较大学习率
        # 这样可以保证梯度流通，同时让 probe 学得更快
        
        # 配置训练参数
        training_args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(self.output_dir, "phase1"),
            num_train_epochs=self.phase1_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.base_lr,  # 这只是默认学习率，我们会用参数分组覆盖
            fp16=True,
            bf16=False,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self.save_total_limit if hasattr(self, 'save_total_limit') and self.save_total_limit else 0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.1,
            dataloader_num_workers=4,
        )
        
        # 基础损失
        base_loss = losses.Contrastive(model=self.base_model)
        
        # IGP 损失
        igp_loss = IGPLoss(
            base_loss=base_loss,
            probe=self.probe,
            adapter=self.adapter,
            gate=self.gate,
            aux_loss_weight=self.aux_loss_weight,
        )
        
        # 使用参数分组学习率：base_model 极小，probe 较大
        param_groups = []
        
        # 基础模型参数 (使用极小学习率)
        base_params = []
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                base_params.append(param)
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.base_lr * 0.01,  # 极小学习率
            })
        
        # Probe 参数 (使用大学习率)
        if self.probe is not None:
            probe_params = list(self.probe.parameters())
            if probe_params:
                param_groups.append({
                    'params': probe_params,
                    'lr': self.base_lr * 10,  # 较大学习率
                })
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # 创建 Phase 1 输出目录
        phase1_output_dir = os.path.join(self.output_dir, "phase1")
        
        # 创建回调
        early_stopping_callback = EarlyStoppingCallback(
            eval_steps=self.eval_steps,
            patience=3,
            threshold=0.25,
        )
        
        loss_tracker_p1 = LossTracker(phase1_output_dir, None, phase_name="phase1")
        igp_modules_p1 = (self.probe, self.adapter, self.gate)
        best_model_callback_p1 = BestModelCallback(phase1_output_dir, self.base_model, phase_name="phase1", igp_modules=igp_modules_p1)
        checkpoint_callback_p1 = EpochCheckpointCallback(phase1_output_dir, self.base_model, phase_name="phase1", igp_modules=igp_modules_p1)
        
        # 创建训练器
        trainer = SentenceTransformerTrainer(
            model=self.base_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=igp_loss,
            data_collator=utils.ColBERTCollator(tokenize_fn=self.base_model.tokenize),
            callbacks=[early_stopping_callback, loss_tracker_p1, best_model_callback_p1, checkpoint_callback_p1],
        )
        trainer.optimizer = optimizer
        
        print("📊 开始 Phase 1 训练...")
        print(f"   基础模型学习率: {self.base_lr * 0.01}")
        print(f"   Probe 学习率: {self.base_lr * 10}")
        print(f"   早停配置: patience=3, threshold=0.25")
        trainer.train()
        
        # 早停检查
        if early_stopping_callback.should_stop:
            best_loss = early_stopping_callback.best_loss
            best_epoch = early_stopping_callback.best_epoch
            last_loss = early_stopping_callback.last_valid_loss
            
            print(f"✅ Phase 1 早停: 满足早停条件")
            if best_loss != float('inf'):
                print(f"   最佳epoch: {best_epoch}, 最佳验证损失: {best_loss:.6f}")
            if last_loss is not None and last_loss != float('inf') and last_loss != best_loss:
                print(f"   最后有效epoch验证损失: {last_loss:.6f}")
        else:
            state = trainer.state
            best_metric = state.best_metric if hasattr(state, 'best_metric') else float('inf')
            print(f"📊 Phase 1 完成: best_eval_loss = {best_metric:.4f}")
        
        return early_stopping_callback.best_loss
    
    def train_phase2(self):
        """
        Phase 2: Joint Training
        
        训练所有参数，门控使用大学习率
        """
        print("\n" + "=" * 60)
        print("🚀 Phase 2: 联合训练 (Joint Training)")
        print("=" * 60)
        
        self.unfreeze_all()
        
        # 配置训练参数
        training_args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(self.output_dir, "phase2"),
            num_train_epochs=self.phase2_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.base_lr,
            fp16=True,
            bf16=False,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=self.save_total_limit if hasattr(self, 'save_total_limit') and self.save_total_limit else 0,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.1,
            dataloader_num_workers=4,
        )
        
        # 基础损失
        base_loss = losses.Contrastive(model=self.base_model)
        
        # IGP 损失
        igp_loss = IGPLoss(
            base_loss=base_loss,
            probe=self.probe,
            adapter=self.adapter,
            gate=self.gate,
            aux_loss_weight=self.aux_loss_weight,
        )
        
        # 创建优化器，为门控分配独立学习率
        param_groups = []
        
        # 基础模型参数 (使用基础学习率)
        base_params = []
        for name, param in self.base_model.named_parameters():
            if param.requires_grad:
                base_params.append(param)
        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.base_lr,
            })
        
        # Probe 参数 (使用基础学习率)
        if self.probe is not None:
            probe_params = list(self.probe.parameters())
            if probe_params:
                param_groups.append({
                    'params': probe_params,
                    'lr': self.base_lr,
                })
        
        # Adapter 参数 (使用基础学习率)
        if self.adapter is not None:
            adapter_params = list(self.adapter.parameters())
            if adapter_params:
                param_groups.append({
                    'params': adapter_params,
                    'lr': self.base_lr,
                })
        
        # Gate 参数 (使用大学习率)
        if self.gate is not None:
            gate_params = list(self.gate.parameters())
            if gate_params:
                param_groups.append({
                    'params': gate_params,
                    'lr': self.gate_lr,  # 大学习率，确保门控能被激活
                })
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # 创建 Phase 2 输出目录
        phase2_output_dir = os.path.join(self.output_dir, "phase2")
        
        # 创建回调
        loss_tracker_p2 = LossTracker(phase2_output_dir, None, phase_name="phase2")
        igp_modules_p2 = (self.probe, self.adapter, self.gate)
        best_model_callback_p2 = BestModelCallback(phase2_output_dir, self.base_model, phase_name="phase2", igp_modules=igp_modules_p2)
        checkpoint_callback_p2 = EpochCheckpointCallback(phase2_output_dir, self.base_model, phase_name="phase2", igp_modules=igp_modules_p2)
        
        phase2_early_stopping = Phase2EarlyStoppingCallback(
            patience=self.phase2_patience,
            threshold=self.phase2_early_stop_threshold,
            mode='min',
        )
        
        # 创建训练器
        trainer = SentenceTransformerTrainer(
            model=self.base_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=igp_loss,
            data_collator=utils.ColBERTCollator(tokenize_fn=self.base_model.tokenize),
            callbacks=[loss_tracker_p2, best_model_callback_p2, phase2_early_stopping, checkpoint_callback_p2],
        )
        trainer.optimizer = optimizer
        
        print(f"📊 开始 Phase 2 训练...")
        print(f"   基础模型学习率: {self.base_lr}")
        print(f"   门控学习率: {self.gate_lr}")
        print(f"   早停配置: patience={self.phase2_patience}, threshold={self.phase2_early_stop_threshold}")
        
        trainer.train()
        
        phase2_best_loss = None
        if hasattr(trainer, 'state') and hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None:
            phase2_best_loss = trainer.state.best_metric
        
        if phase2_early_stopping.should_stop:
            best_metric_val = phase2_early_stopping.best_metric
            best_epoch_val = phase2_early_stopping.best_epoch
            last_metric_val = phase2_early_stopping.last_valid_metric
            
            if best_metric_val == float('inf'):
                print(f"✅ Phase 2 早停: 连续 {self.phase2_patience} 个epoch未改善")
                print(f"   最佳epoch: {best_epoch_val}, 最佳验证损失: N/A (评估未完成)")
                if last_metric_val is not None and last_metric_val != float('inf'):
                    print(f"   最后有效epoch验证损失: {last_metric_val:.6f}")
            else:
                print(f"✅ Phase 2 早停: 连续 {self.phase2_patience} 个epoch未改善")
                print(f"   最佳epoch: {best_epoch_val}, 最佳验证损失: {best_metric_val:.6f}")
                if last_metric_val is not None and last_metric_val != float('inf') and last_metric_val != best_metric_val:
                    print(f"   最后有效epoch验证损失: {last_metric_val:.6f}")
                phase2_best_loss = best_metric_val
        else:
            print("✅ Phase 2 训练完成")
        
        return phase2_best_loss
    
    def train(self):
        """执行完整的两阶段训练"""
        print("=" * 60)
        print("📊 ColBERT-IGP 两阶段训练")
        print("=" * 60)
        
        # Phase 2 独立训练模式验证
        if not self.enable_phase1 and self.enable_phase2:
            print("\n⚠️ Phase 2 独立训练模式")
            if not self.phase1_checkpoint:
                print("\n" + "=" * 60)
                print("❌ 错误: Phase 2 独立训练需要指定 Phase 1 检查点路径")
                print("=" * 60)
                print("请使用 --phase1_checkpoint 参数指定 Phase 1 的检查点路径")
                print("例如: --phase1_checkpoint /path/to/phase1/checkpoint-N")
                raise ValueError("Phase 2 独立训练需要 Phase 1 检查点")
            print(f"✅ Phase 1 检查点已指定: {self.phase1_checkpoint}")
        
        # 加载数据
        self.load_data()
        
        # Phase 1: Probe Warm-up
        if self.enable_phase1:
            print("\n" + "=" * 60)
            print("🔍 Phase 1: Probe Warm-up")
            print("=" * 60)
            
            if self.phase1_checkpoint:
                print(f"\n📂 从检查点恢复 Phase 1: {self.phase1_checkpoint}")
                try:
                    self.load_checkpoint(self.phase1_checkpoint, phase="phase1")
                    print(f"⏭ 已跳过 Phase 1 训练 (从检查点恢复)")
                except Exception as e:
                    print(f"❌ 加载 Phase 1 检查点失败: {str(e)}")
                    raise
                phase1_best_loss = None
            else:
                print("\n⏩ Phase 1 从头开始训练")
                self.load_model()
                phase1_best_loss = self.train_phase1()
        else:
            print("\n⏭ Phase 1 已禁用，跳过")
            phase1_best_loss = None
        
        # Phase 2: Joint Training
        if self.enable_phase2:
            print("\n" + "=" * 60)
            print("🚀 Phase 2: 联合训练")
            print("=" * 60)
            
            if self.phase2_checkpoint:
                print(f"\n📂 从检查点恢复 Phase 2: {self.phase2_checkpoint}")
                try:
                    self.load_checkpoint(self.phase2_checkpoint, phase="phase2")
                    print(f"⏭ 已跳过 Phase 2 训练 (从检查点恢复)")
                except Exception as e:
                    print(f"❌ 加载 Phase 2 检查点失败: {str(e)}")
                    raise
                phase2_best_loss = None
            else:
                # 检查是否需要从 Phase 1 检查点加载
                if not self.enable_phase1 or self.phase1_checkpoint:
                    # Phase 1 已完成或跳过，需要加载 Phase 1 的最终模型
                    print("\n📂 加载 Phase 1 最终模型用于 Phase 2 训练")
                    if self.phase1_checkpoint:
                        phase1_model_path = self.phase1_checkpoint
                    else:
                        # 查找最新的 phase1 检查点
                        phase1_dir = os.path.join(self.output_dir, "phase1")
                        if os.path.exists(phase1_dir):
                            checkpoints = [d for d in os.listdir(phase1_dir) if d.startswith("checkpoint-")]
                            if checkpoints:
                                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
                                phase1_model_path = os.path.join(phase1_dir, checkpoints[-1])
                            else:
                                phase1_model_path = None
                        else:
                            phase1_model_path = None
                    
                    if phase1_model_path and os.path.exists(phase1_model_path):
                        try:
                            self.load_checkpoint(phase1_model_path, phase="phase1")
                            print(f"✅ 已加载 Phase 1 模型: {phase1_model_path}")
                        except Exception as e:
                            print(f"⚠️ 加载 Phase 1 模型失败，使用当前模型: {str(e)}")
                
                # 如果模型还没加载，初始化模型
                if self.base_model is None:
                    self.load_model()
                
                print("\n⏩ Phase 2 从当前模型开始训练")
                phase2_best_loss = self.train_phase2()
        else:
            print("\n⏭ Phase 2 已禁用，跳过")
            phase2_best_loss = None
        
        # 保存最终模型和元数据
        self._save_final_model(phase1_best_loss, phase2_best_loss)
        
        print("\n" + "=" * 60)
        print("✅ 两阶段训练完成!")
        print("=" * 60)
    
    def _save_igp_modules_to_dir(self, save_dir, phase_name="training"):
        """保存 IGP 模块参数到指定目录"""
        probe, adapter, gate = self.probe, self.adapter, self.gate
        
        igp_state = {}
        
        if probe is not None and hasattr(probe, 'state_dict'):
            probe_path = os.path.join(save_dir, "igp_probe.pt")
            torch.save(probe.state_dict(), probe_path)
            igp_state['probe'] = 'igp_probe.pt'
            print(f"   ✅ Probe 参数已保存: {probe_path}")
        
        if adapter is not None and hasattr(adapter, 'state_dict'):
            adapter_path = os.path.join(save_dir, "igp_adapter.pt")
            torch.save(adapter.state_dict(), adapter_path)
            igp_state['adapter'] = 'igp_adapter.pt'
            print(f"   ✅ Adapter 参数已保存: {adapter_path}")
        
        if gate is not None and hasattr(gate, 'state_dict'):
            gate_path = os.path.join(save_dir, "igp_gate.pt")
            torch.save(gate.state_dict(), gate_path)
            igp_state['gate'] = 'igp_gate.pt'
            print(f"   ✅ Gate 参数已保存: {gate_path}")
        
        if igp_state:
            igp_info = {
                'phase': phase_name,
                'modules': igp_state,
                'config': {
                    'enable_probe': probe is not None,
                    'enable_adapter': adapter is not None,
                    'enable_gate': gate is not None,
                }
            }
            igp_info_path = os.path.join(save_dir, "igp_info.json")
            with open(igp_info_path, 'w') as f:
                json.dump(igp_info, f, indent=2)
            print(f"   ✅ IGP 配置已保存: {igp_info_path}")
    
    def _save_final_model(self, phase1_best_loss=None, phase2_best_loss=None):
        """保存最终模型和元数据"""
        print("\n" + "=" * 60)
        print("💾 保存最终模型")
        print("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        final_model_dir = os.path.join(self.output_dir, f"final_model_{timestamp}")
        os.makedirs(final_model_dir, exist_ok=True)
        
        self.base_model.save(final_model_dir)
        print(f"✅ 最终模型已保存: {final_model_dir}")
        
        self._save_igp_modules_to_dir(final_model_dir, phase_name="final")
        
        metadata = {
            "model_name": self.model_name,
            "save_timestamp": timestamp,
            "phase1_best_loss": phase1_best_loss,
            "phase2_best_loss": phase2_best_loss,
            "phase1_epochs": self.phase1_epochs,
            "phase2_epochs": self.phase2_epochs,
            "batch_size": self.batch_size,
            "base_lr": self.base_lr,
            "gate_lr": self.gate_lr,
            "enable_probe": self.enable_probe,
            "enable_adapter": self.enable_adapter,
            "enable_gate": self.enable_gate,
            "max_ratio": self.max_ratio,
            "bottleneck_dim": self.bottleneck_dim,
            "aux_loss_weight": self.aux_loss_weight,
        }
        
        metadata_file = os.path.join(final_model_dir, "training_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"✅ 训练元数据已保存: {metadata_file}")
        
        summary_file = os.path.join(self.output_dir, "training_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ColBERT-IGP 训练总结\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"保存时间: {timestamp}\n\n")
            f.write("训练阶段:\n")
            if phase1_best_loss is not None and phase1_best_loss != float('inf'):
                f.write(f"  Phase 1 (Probe Warm-up): {self.phase1_epochs} epochs, 最佳验证损失: {phase1_best_loss:.6f}\n")
            else:
                f.write(f"  Phase 1: 已跳过\n")
            if phase2_best_loss is not None and phase2_best_loss != float('inf'):
                f.write(f"  Phase 2 (Joint Training): {self.phase2_epochs} epochs, 最佳验证损失: {phase2_best_loss:.6f}\n")
            else:
                f.write(f"  Phase 2: 已跳过\n")
            f.write("\n模型配置:\n")
            f.write(f"  启用 Probe: {self.enable_probe}\n")
            f.write(f"  启用 Adapter: {self.enable_adapter}\n")
            f.write(f"  启用 Gate: {self.enable_gate}\n")
            f.write(f"  Max Ratio: {self.max_ratio}\n")
            f.write(f"  Bottleneck Dim: {self.bottleneck_dim}\n")
            f.write("\n超参数:\n")
            f.write(f"  Batch Size: {self.batch_size}\n")
            f.write(f"  Base LR: {self.base_lr}\n")
            f.write(f"  Gate LR: {self.gate_lr}\n")
            f.write(f"  Aux Loss Weight: {self.aux_loss_weight}\n")
            f.write("\n模型文件:\n")
            f.write(f"  最终模型: {final_model_dir}\n")
        print(f"✅ 训练总结已保存: {summary_file}")


def load_followir_train_data(data_path):
    """加载 FollowIR 训练数据"""
    print(f"📂 加载训练数据: {data_path}")
    
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = eval(line.strip())
            
            query = item.get('query', '')
            positives = item.get('pos', [])
            negatives = item.get('neg', [])
            
            if positives and negatives:
                for pos_doc in positives:
                    for neg_doc in negatives:
                        data_list.append({
                            'query': query,
                            'document': pos_doc,
                            'negative': neg_doc,
                        })
    
    print(f"✅ 加载了 {len(data_list)} 个训练样本")
    return Dataset.from_list(data_list)


def main():
    parser = argparse.ArgumentParser(description='ColBERT-IGP 两阶段训练')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, 
                       default='lightonai/GTE-ModernColBERT-v1',
                       help='基座模型名称或路径')
    
    # 训练数据参数
    parser.add_argument('--train_data', type=str, 
                       default='/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/colbert_train_final.jsonl',
                       help='训练数据路径')
    parser.add_argument('--output_dir', type=str, 
                       default='/home/luwa/Documents/pylate/output/colbert_igp_train',
                       help='输出目录')
    
    # 训练参数
    parser.add_argument('--phase1_epochs', type=int, default=2,
                        help='Phase 1 训练轮数')
    parser.add_argument('--phase2_epochs', type=int, default=3,
                        help='Phase 2 训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='基础学习率')
    parser.add_argument('--gate_lr', type=float, default=1e-2,
                        help='门控学习率')
    
    # 训练阶段控制参数
    parser.add_argument('--enable_phase1', type=lambda x: x.lower() == 'true', 
                       default=True, help='启用 Phase 1 (Probe Warm-up)')
    parser.add_argument('--enable_phase2', type=lambda x: x.lower() == 'true', 
                       default=True, help='启用 Phase 2 (Joint Training)')
    
    # IGP 模块参数
    parser.add_argument('--enable_probe', type=lambda x: x.lower() == 'true', 
                       default=True, help='启用 InstructionProbe')
    parser.add_argument('--enable_adapter', type=lambda x: x.lower() == 'true', 
                       default=True, help='启用 IGPAdapter')
    parser.add_argument('--enable_gate', type=lambda x: x.lower() == 'true', 
                       default=True, help='启用 RatioGate')
    parser.add_argument('--max_ratio', type=float, default=0.2,
                        help='门控最大比率')
    parser.add_argument('--bottleneck_dim', type=int, default=64,
                        help='Adapter 瓶颈维度')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='辅助损失权重')
    
    # Phase 2 早停参数
    parser.add_argument('--phase2_patience', type=int, default=3,
                        help='Phase 2 早停 patience (连续多少个epoch未改善则停止)')
    parser.add_argument('--phase2_early_stop_threshold', type=float, default=0.001,
                        help='Phase 2 早停阈值 (改善的最小量)')
    
    # 检查点保存配置
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='最多保存的检查点数量 (默认=phase1_epochs + phase2_epochs)')
    
    # 检查点配置
    parser.add_argument('--phase1_checkpoint', type=str, default=None,
                        help='Phase 1 检查点路径 (从指定检查点恢复，或为空则从头训练)')
    parser.add_argument('--phase2_checkpoint', type=str, default=None,
                        help='Phase 2 检查点路径 (从指定检查点恢复，或为空则从头训练)')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU 设备')
    parser.add_argument('--note', type=str, default='',
                        help='备注信息')
    
    args = parser.parse_args()
    
    # 如果未指定 save_total_limit，默认值为总 epoch 数
    if args.save_total_limit is None:
        total_epochs = args.phase1_epochs + args.phase2_epochs
        args.save_total_limit = max(total_epochs, 3)  # 至少保留3个
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]
    
    # 创建输出目录 - 直接使用用户指定的路径
    output_dir = args.output_dir
    
    # 如果输出目录已有内容，先清空（支持重新训练）
    if os.path.exists(output_dir) and os.listdir(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        print(f"🗑️ 已清空已有输出目录: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("📊 ColBERT-IGP 两阶段训练")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"训练数据: {args.train_data}")
    print(f"输出目录: {output_dir}")
    print(f"Phase 1 epochs: {args.phase1_epochs}")
    print(f"Phase 2 epochs: {args.phase2_epochs}")
    print(f"检查点最大保存数: {args.save_total_limit}")
    print(f"批次大小: {args.batch_size}")
    print(f"基础学习率: {args.base_lr}")
    print(f"门控学习率: {args.gate_lr}")
    print(f"启用 Probe: {args.enable_probe}")
    print(f"启用 Adapter: {args.enable_adapter}")
    print(f"启用 Gate: {args.enable_gate}")
    print("=" * 60)
    
    # 保存参数
    params_file = os.path.join(output_dir, "training_params.txt")
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ColBERT-IGP 两阶段训练参数\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"训练数据: {args.train_data}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"Phase 1 epochs: {args.phase1_epochs}\n")
        f.write(f"Phase 2 epochs: {args.phase2_epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"基础学习率: {args.base_lr}\n")
        f.write(f"门控学习率: {args.gate_lr}\n")
        f.write(f"启用 Probe: {args.enable_probe}\n")
        f.write(f"启用 Adapter: {args.enable_adapter}\n")
        f.write(f"启用 Gate: {args.enable_gate}\n")
        f.write(f"门控最大比率: {args.max_ratio}\n")
        f.write(f"Adapter瓶颈维度: {args.bottleneck_dim}\n")
        f.write(f"辅助损失权重: {args.aux_loss_weight}\n")
        f.write(f"GPU设备: {args.device}\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.note:
            f.write(f"备注: {args.note}\n")
        f.write("=" * 60 + "\n")
    print(f"📝 参数已保存至: {params_file}")
    
    # 创建训练器并训练
    trainer = IGPTrainer(
        model_name=args.model_name,
        train_data_path=args.train_data,
        output_dir=output_dir,
        device=args.device,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        batch_size=args.batch_size,
        eval_ratio=args.eval_ratio,
        base_lr=args.base_lr,
        gate_lr=args.gate_lr,
        eval_steps=50,
        enable_phase1=args.enable_phase1,
        enable_phase2=args.enable_phase2,
        enable_probe=args.enable_probe,
        enable_adapter=args.enable_adapter,
        enable_gate=args.enable_gate,
        max_ratio=args.max_ratio,
        bottleneck_dim=args.bottleneck_dim,
        aux_loss_weight=args.aux_loss_weight,
        phase2_patience=args.phase2_patience,
        phase2_early_stop_threshold=args.phase2_early_stop_threshold,
        save_total_limit=args.save_total_limit,
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
