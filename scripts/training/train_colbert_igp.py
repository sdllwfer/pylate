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
print(f"[DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}")
print(f"[DEBUG] torch.cuda.current_device() = {torch.cuda.current_device()}")
print(f"[DEBUG] os.environ.get('CUDA_VISIBLE_DEVICES') = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
import torch.nn as nn
import torch.nn.functional as F
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
    InstructionProbeV2,
    IGPAdapterV2,
    RatioGateV2,
    RatioGateV3,
    IGPColBERTWrapper,
)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import DataLoader, DataConverter, IGPColBERTCollator
from igp_losses import IGPAuxLoss, IGPLoss


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
        if best == float('inf'):
            return True
        if self.threshold_mode == 'rel':
            return current < best - best * self.threshold
        else:
            return current < best - self.threshold
    
    def _is_better_max(self, current, best):
        if best == float('-inf'):
            return True
        if self.threshold_mode == 'rel':
            return current > best + best * self.threshold
        else:
            return current > best + self.threshold
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        
        current_metric = metrics.get('eval_loss', float('inf'))
        epoch = int(state.epoch) if state.epoch else 0
        
        print(f"\n[DEBUG] Phase2EarlyStopping - epoch={epoch}, current_metric={current_metric}, best_metric={self.best_metric}, threshold={self.threshold}")
        
        # 第一次评估时，如果 best_metric 是 inf，直接更新
        if self.best_metric == float('inf') and current_metric != float('inf'):
            print(f"[DEBUG] 第一次有效评估，更新 best_metric")
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.no_improve_count = 0
            self.last_valid_metric = current_metric
            return
        
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


class IGPColBERTTrainer(SentenceTransformerTrainer):
    """自定义 IGP ColBERT Trainer
    
    确保训练和验证时都使用 IGP 模块计算损失
    """
    
    def __init__(self, *args, loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        # 保存自定义损失函数到 self.loss，覆盖父类的损失
        if loss is not None:
            self.loss = loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """计算损失 - 训练和验证都使用 IGP 损失"""
        # 使用父类的 collect_features 方法处理输入
        features, labels = self.collect_features(inputs)
        loss = self.loss(features, labels)
        return (loss, {}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """预测步骤 - 确保验证时使用 IGP 损失"""
        with torch.no_grad():
            # 使用父类的 collect_features 方法处理输入
            features, labels = self.collect_features(inputs)
            loss = self.loss(features, labels)
            return (loss, None, None)


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
    
    def __init__(self, output_dir, trainer, phase_name="training", log_interval=10):
        """
        初始化损失追踪器
        
        Args:
            output_dir: 输出目录
            trainer: 训练器实例
            phase_name: 阶段名称 (phase1/phase2)
            log_interval: 损失记录间隔 (每多少个step记录一次)
        """
        self.output_dir = output_dir
        self.trainer = trainer
        self.phase_name = phase_name
        self.log_interval = log_interval
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.batch_losses = []
        self.sampled_losses = []
        self.current_step = 0
        self.epochs = []
        self.current_train_loss = None
        self.history_file = os.path.join(output_dir, "loss_history.json")
        
        # 新增：各项损失的追踪
        self.rank_losses = []
        self.aux_losses = []
        self.reg_losses = []
        self.gate_ratios = []
        
        # 新增：训练 loss 累积变量（用于计算 epoch 平均）
        self.epoch_loss_sum = 0.0
        self.epoch_loss_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                loss_val = logs.get('loss')
                self.current_train_loss = loss_val
                # 累积 loss 用于计算 epoch 平均
                if torch.is_tensor(loss_val):
                    self.epoch_loss_sum += loss_val.item()
                else:
                    self.epoch_loss_sum += loss_val
                self.epoch_loss_count += 1
                
                if 'step' in logs:
                    step = logs.get('step', self.current_step)
                    self.batch_losses.append({
                        'step': step,
                        'train_loss': self.current_train_loss,
                        'phase': self.phase_name,
                        # 记录各项损失
                        'rank_loss': logs.get('rank_loss', 0.0),
                        'aux_loss': logs.get('aux_loss', 0.0),
                        'reg_loss': logs.get('reg_loss', 0.0),
                        'gate_ratio': logs.get('gate_ratio', 0.0),
                    })
                    if self.log_interval > 0 and step % self.log_interval == 0:
                        self.sampled_losses.append({
                            'step': step,
                            'train_loss': self.current_train_loss,
                            'phase': self.phase_name
                        })
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.current_step = 0
        self.batch_losses = []
        self.sampled_losses = []
    
    def on_step_end(self, args, state, control, outputs=None, **kwargs):
        if outputs is not None and hasattr(outputs, 'loss'):
            loss = outputs.loss.item() if hasattr(outputs.loss, 'item') else outputs.loss
            self.current_step = state.global_step
            
            self.batch_losses.append({
                'step': self.current_step,
                'train_loss': loss,
                'phase': self.phase_name
            })
            
            if self.log_interval > 0 and self.current_step % self.log_interval == 0:
                self.sampled_losses.append({
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
            # 使用 epoch 平均 loss 而不是最后一个 logging step 的 loss
            if self.epoch_loss_count > 0:
                epoch_avg_loss = self.epoch_loss_sum / self.epoch_loss_count
                self.train_losses.append(epoch_avg_loss)
                self.train_steps.append(state.global_step)
                print(f"\n📊 Epoch {epoch} 平均训练损失: {epoch_avg_loss:.6f} (基于 {self.epoch_loss_count} 个 batch)")
            elif self.current_train_loss is not None:
                self.train_losses.append(self.current_train_loss)
                self.train_steps.append(state.global_step)
            # 重置累积变量
            self.epoch_loss_sum = 0.0
            self.epoch_loss_count = 0
            self.current_train_loss = None
        
        self._save_history()
        self._plot_losses()
    
    def _save_history(self):
        # 确保目录存在（防止目录被意外删除）
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        # 确保目录存在（防止目录被意外删除）
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        # 确保目录存在（防止目录被意外删除）
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.sampled_losses:
            sampled_steps = [b['step'] for b in self.sampled_losses]
            sampled_losses = [b['train_loss'] for b in self.sampled_losses]
        else:
            sampled_steps = [b['step'] for b in self.batch_losses]
            sampled_losses = [b['train_loss'] for b in self.batch_losses]
        
        eval_steps = [s for s in self.eval_steps if s is not None]
        eval_losses_filtered = [l for l in self.eval_losses if l is not None]
        
        if not sampled_losses and not eval_losses_filtered:
            return
        
        self._plot_step_curve(sampled_steps, sampled_losses, eval_steps, eval_losses_filtered)
        
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
    """自定义回调：每个epoch完成后保存checkpoint，按epoch编号命名
    
    检查点保存到 checkpoints/ 子目录中
    """
    
    def __init__(self, output_dir, model, phase_name="training", igp_modules=None):
        self.output_dir = output_dir
        self.model = model
        self.phase_name = phase_name
        self.igp_modules = igp_modules
        self.saved_checkpoints = []
        # 创建 checkpoints 子目录
        self.checkpoints_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch else 0
        if epoch > 0:
            # 保存到 checkpoints/ 子目录
            checkpoint_dir = os.path.join(self.checkpoints_dir, f"checkpoint-{epoch}")
            os.makedirs(checkpoint_dir, exist_ok=True)
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


# class IGPLoss(nn.Module):
#     """
#     IGP 组合损失
    
#     结合对比损失和辅助损失 (BCE Loss for instruction detection)
#     同时应用 Adapter 和 Gate 到模型输出
    
#     注意: 此类作为 SentenceTransformerTrainer 的 loss 接口，
#     需要返回单个标量损失值。
#     """
    
#     def __init__(
#         self,
#         base_loss,
#         base_model=None,
#         probe: Optional[InstructionProbe] = None,
#         adapter: Optional[IGPAdapter] = None,
#         gate: Optional[RatioGate] = None,
#         aux_loss_weight: float = 0.1,
#     ):
#         super().__init__()
#         self.base_loss = base_loss
#         self.base_model = base_model
#         self.probe = probe
#         self.adapter = adapter
#         self.gate = gate
#         self.aux_loss_weight = aux_loss_weight
#         self.aux_loss_fn = IGPAuxLoss(pos_weight=10.0)
#         self.rank_loss_fn = nn.CrossEntropyLoss()
#         self.temperature = 1.0  # 温度参数
    
#     def forward(self, sentence_features: list, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         参照参考实现 colbert_igp.py 的完整逻辑：
#         1. 获取 Q_origin (ColBERT embeddings)
#         2. Probe 提取 inst_vec
#         3. Adapter 计算 delta
#         4. shift_mask 锁定 Query 部分
#         5. 弹性门控缩放
#         6. 归一化并计算分数
#         """
#         # sentence_features: [query_features, pos_features, neg_features]
#         query_features = sentence_features[0]
#         pos_features = sentence_features[1]
#         neg_features = sentence_features[2]
        
#         query_input_ids = query_features.get('input_ids')
#         query_attention_mask = query_features.get('attention_mask')
#         instruction_mask = query_features.get('instruction_mask')
        
#         # ========== 1. 获取 ColBERT embeddings (Q_origin) ==========
#         # 注意：使用 ColBERT 的投影输出，不是原始 word embeddings
#         # Phase 1: 不使用 no_grad，让梯度流向 probe
#         # Phase 2: 依赖参数冻结来阻止 base_model 更新
#         if query_input_ids is not None and self.base_model is not None:
#             query_word_emb = self.base_model[0].auto_model.embeddings(query_input_ids)
#             pos_word_emb = self.base_model[0].auto_model.embeddings(pos_features.get('input_ids'))
#             neg_word_emb = self.base_model[0].auto_model.embeddings(neg_features.get('input_ids'))
            
#             query_word_emb = query_word_emb * query_attention_mask.unsqueeze(-1).float()
#             pos_word_emb = pos_word_emb * pos_features.get('attention_mask').unsqueeze(-1).float()
#             neg_word_emb = neg_word_emb * neg_features.get('attention_mask').unsqueeze(-1).float()
            
#             # 投影到 ColBERT 空间得到 Q_origin
#             if hasattr(self.base_model[0].auto_model, 'projector'):
#                 Q_origin = self.base_model[0].auto_model.projector(query_word_emb)
#                 pos_colbert = self.base_model[0].auto_model.projector(pos_word_emb)
#                 neg_colbert = self.base_model[0].auto_model.projector(neg_word_emb)
#             else:
#                 Q_origin = query_word_emb
#                 pos_colbert = pos_word_emb
#                 neg_colbert = neg_word_emb
#         else:
#             Q_origin = query_features.get('token_embeddings', query_features.get('embeddings'))
#             pos_colbert = pos_features.get('token_embeddings', pos_features.get('embeddings'))
#             neg_colbert = neg_features.get('token_embeddings', neg_features.get('embeddings'))
        
#         # ========== 2. Probe: 提取指令向量 ==========
#         # 使用 Q_origin 而不是原始 word embeddings
#         inst_vec = None
#         attn_logits = None
#         if self.probe is not None:
#             inst_vec, attn_logits, _ = self.probe(Q_origin, query_attention_mask)
        
#         # ========== 3. 辅助损失: 指令检测 ==========
#         aux_loss = torch.tensor(0.0, device=Q_origin.device)
#         if instruction_mask is not None and query_attention_mask is not None and attn_logits is not None:
#             # 对齐 mask 长度 (ColBERT 可能会去掉 CLS)
#             seq_len = Q_origin.shape[1]
#             if instruction_mask.shape[1] > seq_len:
#                 target_mask = instruction_mask[:, 1:1+seq_len]
#             else:
#                 target_mask = instruction_mask[:, :seq_len]
#             target_mask = target_mask[:, :seq_len].float()
            
#             # 计算 BCE Loss
#             aux_loss = self.aux_loss_fn.compute(attn_logits, target_mask, query_attention_mask)
        
#         # ========== 4. Adapter: 计算 delta ==========
#         # 逻辑: [Query, Inst] 拼接 -> delta
#         delta = None
#         Q_final = Q_origin
#         if self.adapter is not None and inst_vec is not None:
#             # 扩展 inst_vec 到序列维度
#             inst_expanded = inst_vec.unsqueeze(1).expand(-1, Q_origin.size(1), -1)
#             # 拼接: [B, Seq, Dim*2]
#             combined = torch.cat([Q_origin, inst_expanded], dim=-1)
#             # 使用 down_project -> activation -> up_project
#             bottleneck = F.gelu(self.adapter.down_project(combined))
#             bottleneck = self.adapter.dropout(bottleneck)
#             delta = self.adapter.up_project(bottleneck)  # [B, Seq, Dim]
        
#         # ========== 5. Gate: 弹性门控 ==========
#         gate_ratio = 0.0
#         if self.gate is not None and delta is not None and instruction_mask is not None:
#             # 生成 shift_mask: 锁定 Query 部分 (非指令部分)
#             # Query(1) - Instruction(1) = PureQuery(1)
#             raw_mask = (query_attention_mask.float() - instruction_mask.float()).clamp(min=0)
            
#             # 对齐 Q_origin 长度
#             seq_len_q = Q_origin.shape[1]
#             seq_len_mask = raw_mask.shape[1]
#             if seq_len_q == seq_len_mask - 1:
#                 align_mask = raw_mask[:, 1:]
#             else:
#                 min_len = min(seq_len_q, seq_len_mask)
#                 align_mask = raw_mask[:, :min_len]
            
#             shift_mask = align_mask.unsqueeze(-1).to(Q_origin.device)  # [B, Seq, 1]
            
#             # 计算 Query 范数
#             q_vec = Q_origin * shift_mask
#             q_norm = torch.norm(q_vec, p=2, dim=-1, keepdim=True) + 1e-8
            
#             # Delta 单位方向
#             delta_unit = F.normalize(delta, p=2, dim=-1)
            
#             # 弹性比例: ratio * 0.7 (max_ratio)
#             if hasattr(self.gate, 'max_ratio'):
#                 max_ratio = self.gate.max_ratio
#             else:
#                 max_ratio = 0.7
            
#             current_ratio = torch.sigmoid(self.gate.ratio_gate) * max_ratio if hasattr(self.gate, 'ratio_gate') else max_ratio * 0.5
#             gate_ratio = current_ratio.item()
            
#             # 弹性偏移: 方向 * (Q长度 * 比例)
#             effective_delta = delta_unit * (q_norm * current_ratio)
#             effective_delta = effective_delta * shift_mask
            
#             # Q_hat = Q_origin + effective_delta
#             Q_hat = Q_origin + effective_delta
            
#             # 归一化
#             Q_final = F.normalize(Q_hat, p=2, dim=-1)
        
#         # ========== 6. 计算 Ranking Loss ==========
#         pos_scores = self._colbert_maxsim(Q_final, query_attention_mask, pos_colbert, pos_features.get('attention_mask'))
#         neg_scores = self._colbert_maxsim(Q_final, query_attention_mask, neg_colbert, neg_features.get('attention_mask'))
        
#         # 使用 temperature 缩放 logits
#         pos_scores_scaled = pos_scores / self.temperature
#         neg_scores_scaled = neg_scores / self.temperature
        
#         scores = torch.cat([pos_scores_scaled, neg_scores_scaled], dim=0)
#         rank_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)], dim=0)
#         rank_loss = self.rank_loss_fn(scores, rank_labels)
        
#         # ========== 7. 正则化损失 ==========
#         reg_loss = torch.tensor(0.0, device=rank_loss.device)
#         if delta is not None:
#             delta_norm = torch.norm(delta, p=2, dim=-1)
#             excess = torch.clamp(delta_norm - 1.0, min=0)
#             reg_loss = excess.mean()
        
#         # ========== 8. 总损失 ==========
#         total_loss = rank_loss + aux_loss * self.aux_loss_weight + reg_loss
        
#         # ========== 9. 保存各项损失用于日志 ==========
#         self._last_losses = {
#             'rank_loss': rank_loss.item(),
#             'aux_loss': aux_loss.item(),
#             'reg_loss': reg_loss.item(),
#             'gate_ratio': gate_ratio,
#             'total_loss': total_loss.item(),
#         }
        
#         return total_loss
    
#     def _colbert_maxsim(
#         self,
#         query_emb: torch.Tensor,
#         query_mask: torch.Tensor,
#         doc_emb: torch.Tensor,
#         doc_mask: torch.Tensor,
#     ) -> torch.Tensor:
#         """计算 ColBERT MaxSim 分数（平均值）"""
#         # query_emb: [batch, seq_q, dim]
#         # doc_emb: [batch, seq_d, dim]
        
#         # 计算点积矩阵 [batch, seq_q, seq_d]
#         scores = torch.einsum('bqd,bsd->bqs', query_emb, doc_emb)
        
#         # Mask 无效位置
#         if doc_mask is not None:
#             # 将 doc 中 padding 位置设为 -inf
#             scores = scores.masked_fill(~doc_mask.unsqueeze(1).bool(), -1e9)
        
#         # MaxSim: 对 doc 维度取 max，然后 sum
#         max_scores, _ = scores.max(dim=-1)  # [batch, seq_q]
        
#         # 直接返回 sum，不做归一化
#         return max_scores.sum(dim=-1)  # [batch]
    
#     def get_last_losses(self) -> dict:
#         """获取上一次前向传播的各项损失值"""
#         return getattr(self, '_last_losses', {})

class IGPLossLocal(nn.Module):
    """
    IGP 组合损失 (本地版本)
    
    结合对比损失和辅助损失 (BCE Loss for instruction detection)
    同时应用 Adapter 和 Gate 到模型输出
    """
    
    def __init__(
        self,
        base_loss,
        base_model=None,
        probe: Optional[InstructionProbe] = None,
        adapter: Optional[IGPAdapter] = None,
        gate: Optional[RatioGate] = None,
        aux_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.base_model = base_model
        self.probe = probe
        self.adapter = adapter
        self.gate = gate
        self.aux_loss_weight = aux_loss_weight
        self.aux_loss_fn = IGPAuxLoss(pos_weight=10.0)
        self.rank_loss_fn = nn.CrossEntropyLoss()
        self.temperature = 0.1  # 用于缩放 logits
    
    def forward(self, sentence_features: list, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(sentence_features, dict):
            query_features = {
                'input_ids': sentence_features.get('input_ids'),
                'attention_mask': sentence_features.get('attention_mask'),
                'token_labels': sentence_features.get('token_labels'),
            }
            pos_features = {
                'input_ids': sentence_features.get('positive_input_ids'),
                'attention_mask': sentence_features.get('positive_attention_mask'),
            }
            neg_features = {
                'input_ids': sentence_features.get('negative_input_ids'),
                'attention_mask': sentence_features.get('negative_attention_mask'),
            }
        else:
            query_features = sentence_features[0]
            pos_features = sentence_features[1]
            neg_features = sentence_features[2]
        
        query_attention_mask = query_features.get('attention_mask')
        instruction_mask = query_features.get('instruction_mask') or query_features.get('token_labels')
        
        # ========== 1. 获取基础表示 (保持 768 维) ==========
        # 【修复维度报错】：获取大模型 Transformer 层的输出 (768维)，而不是投影后的 128维
        query_out = self.base_model[0](query_features)
        pos_out = self.base_model[0](pos_features)
        neg_out = self.base_model[0](neg_features)
        
        Q_hidden = query_out['token_embeddings']  # 768维
        pos_hidden = pos_out['token_embeddings']
        neg_hidden = neg_out['token_embeddings']
        
        # ========== 2. Probe: 提取指令向量 (操作在 768 维) ==========
        inst_vec = None
        attn_logits = None
        if self.probe is not None:
            inst_vec, attn_logits, _ = self.probe(Q_hidden, query_attention_mask)
        
        # ========== 3. 辅助损失: 指令检测 ==========
        aux_loss = torch.tensor(0.0, device=Q_hidden.device, requires_grad=True)
        if instruction_mask is not None and query_attention_mask is not None and attn_logits is not None:
            seq_len = Q_hidden.shape[1]
            if instruction_mask.shape[1] > seq_len:
                target_mask = instruction_mask[:, 1:1+seq_len]
            else:
                target_mask = instruction_mask[:, :seq_len]
            target_mask = target_mask[:, :seq_len].float()
            
            aux_loss = self.aux_loss_fn.compute(attn_logits, target_mask, query_attention_mask)
        
        # ========== 4. Adapter: 计算 delta (操作在 768 维) ==========
        delta = None
        Q_hat = Q_hidden
        if self.adapter is not None and inst_vec is not None:
            inst_expanded = inst_vec.unsqueeze(1).expand(-1, Q_hidden.size(1), -1)
            combined = torch.cat([Q_hidden, inst_expanded], dim=-1)
            bottleneck = F.gelu(self.adapter.down_project(combined))
            bottleneck = self.adapter.dropout(bottleneck)
            delta = self.adapter.up_project(bottleneck)
        
        # ========== 5. Gate: 弹性门控 (操作在 768 维) ==========
        gate_ratio = 0.0
        if self.gate is not None and delta is not None and instruction_mask is not None:
            raw_mask = (query_attention_mask.float() - instruction_mask.float()).clamp(min=0)
            
            seq_len_q = Q_hidden.shape[1]
            seq_len_mask = raw_mask.shape[1]
            if seq_len_q == seq_len_mask - 1:
                align_mask = raw_mask[:, 1:]
            else:
                min_len = min(seq_len_q, seq_len_mask)
                align_mask = raw_mask[:, :min_len]
            
            shift_mask = align_mask.unsqueeze(-1).to(Q_hidden.device)
            
            q_vec = Q_hidden * shift_mask
            q_norm = torch.norm(q_vec, p=2, dim=-1, keepdim=True) + 1e-8
            
            delta_unit = F.normalize(delta, p=2, dim=-1)
            
            max_ratio = self.gate.max_ratio if hasattr(self.gate, 'max_ratio') else 0.7
            current_ratio = torch.sigmoid(self.gate.ratio_gate) * max_ratio if hasattr(self.gate, 'ratio_gate') else max_ratio * 0.5
            gate_ratio = current_ratio.item()
            
            effective_delta = delta_unit * (q_norm * current_ratio)
            effective_delta = effective_delta * shift_mask
            
            # 在 768 维空间完成指令特征的融合
            Q_hat = Q_hidden + effective_delta
        
        # ========== 6. 投影到 ColBERT 维度并归一化 (768维 -> 128维) ==========
        if hasattr(self.base_model[0].auto_model, 'projector'):
            Q_final = self.base_model[0].auto_model.projector(Q_hat)
            pos_colbert = self.base_model[0].auto_model.projector(pos_hidden)
            neg_colbert = self.base_model[0].auto_model.projector(neg_hidden)
        else:
            Q_final = Q_hat
            pos_colbert = pos_hidden
            neg_colbert = neg_hidden
            
        # ColBERT 必须的 L2 归一化
        Q_final = F.normalize(Q_final, p=2, dim=-1)
        pos_colbert = F.normalize(pos_colbert, p=2, dim=-1)
        neg_colbert = F.normalize(neg_colbert, p=2, dim=-1)
        
        # ========== 7. 计算 Ranking Loss ==========
        pos_scores = self._colbert_maxsim(Q_final, query_attention_mask, pos_colbert, pos_features.get('attention_mask'))
        neg_scores = self._colbert_maxsim(Q_final, query_attention_mask, neg_colbert, neg_features.get('attention_mask'))
        
        pos_scores_scaled = pos_scores / self.temperature
        neg_scores_scaled = neg_scores / self.temperature
        
        scores = torch.stack([pos_scores_scaled, neg_scores_scaled], dim=1)
        rank_labels = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        rank_loss = self.rank_loss_fn(scores, rank_labels)
        
        # ========== 8. 正则化损失 ==========
        reg_loss = torch.tensor(0.0, device=rank_loss.device, requires_grad=True)
        if delta is not None:
            delta_norm = torch.norm(delta, p=2, dim=-1)
            excess = torch.clamp(delta_norm - 1.0, min=0)
            reg_loss = excess.mean()
        
        # ========== 9. 决定总损失 ==========
        is_phase1 = False
        if self.adapter is not None:
            is_phase1 = not next(self.adapter.parameters()).requires_grad
        
        # 确保损失连接到计算图（即使值为0）
        if is_phase1:
            # Phase 1: 如果 aux_loss 为0，使用一个虚拟损失确保梯度流动
            # 注意：不能使用 .item()，否则会断开计算图
            if inst_vec is not None:
                # 使用 inst_vec 的 L2 范数作为虚拟损失，确保 probe 有梯度
                dummy_loss = torch.norm(inst_vec, p=2).mean() * 0.01
                total_loss = aux_loss + dummy_loss
            else:
                total_loss = aux_loss
        else:
            total_loss = rank_loss + aux_loss * self.aux_loss_weight + reg_loss
        
        self._last_losses = {
            'rank_loss': rank_loss.item(),
            'aux_loss': aux_loss.item(),
            'reg_loss': reg_loss.item(),
            'gate_ratio': gate_ratio,
            'total_loss': total_loss.item(),
            'phase_mode': 'Phase 1' if is_phase1 else 'Phase 2'
        }
        
        return total_loss

    def _colbert_maxsim(
        self,
        query_emb: torch.Tensor,
        query_mask: torch.Tensor,
        doc_emb: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 ColBERT MaxSim 分数"""
        scores = torch.einsum('bqd,bsd->bqs', query_emb, doc_emb)
        
        if doc_mask is not None:
            scores = scores.masked_fill(~doc_mask.unsqueeze(1).bool(), -1e9)
        
        max_scores, _ = scores.max(dim=-1) 
        
        if query_mask is not None:
            max_scores = max_scores * query_mask.float()
        
        return max_scores.sum(dim=-1)
    
    def get_last_losses(self) -> dict:
        return getattr(self, '_last_losses', {})
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
        probe_num_layers: int = 2,
        aux_loss_weight: float = 0.1,
        gate_l1_coeff: float = 0.01,  # 兼容旧参数，实际不再使用
        lambda_gate: float = 1.0,  # 门控监督损失权重
        phase2_patience: int = 3,
        phase2_early_stop_threshold: float = 0.001,
        save_total_limit: int = 3,
        phase1_checkpoint: str = None,
        phase2_checkpoint: str = None,
        log_interval: int = 10,
        gradient_accumulation_steps: int = 1,
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
        self.probe_num_layers = probe_num_layers
        self.aux_loss_weight = aux_loss_weight
        self.gate_l1_coeff = gate_l1_coeff  # 兼容旧参数，实际不再使用
        self.lambda_gate = lambda_gate  # 门控监督损失权重
        self.phase2_patience = phase2_patience
        self.phase2_early_stop_threshold = phase2_early_stop_threshold
        self.phase1_checkpoint = phase1_checkpoint
        self.phase2_checkpoint = phase2_checkpoint
        self.log_interval = log_interval
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
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
        
        dataset, stats = DataLoader.load_from_file(
            self.train_data_path,
            expand_pairs=True,
            validate=True
        )
        
        print(f"📊 原始数据: {stats.get('total_raw', 0)} 条")
        print(f"📊 展开后: {stats.get('total_expanded', 0)} 条")
        
        dataset = DataLoader.add_instruction_masks(
            dataset,
            tokenizer=self.base_model.tokenizer,
            max_query_length=512,
        )
        
        splits = dataset.train_test_split(test_size=self.eval_ratio)
        self.train_dataset = splits['train']
        self.eval_dataset = splits['test']
        
        print(f"✅ 训练集: {len(self.train_dataset)} 样本")
        print(f"✅ 验证集: {len(self.eval_dataset)} 样本")
        
        sample = self.train_dataset[0] if len(self.train_dataset) > 0 else {}
        if 'combined_query' in sample:
            print(f"📝 示例数据:")
            print(f"   combined_query: {sample['combined_query'][:80]}...")
            if 'instruction_mask' in sample:
                mask = sample['instruction_mask']
                print(f"   instruction_mask: {mask[:10]}... (长度: {len(mask)})")
    
    def load_model(self):
        """加载基础模型"""
        print(f"📥 加载基础模型: {self.model_name} on {self.device}")
        # 使用更大的 query_length 和 document_length 以避免截断
        self.base_model = models.ColBERT(
            model_name_or_path=self.model_name,
            device=self.device,
            query_length=512,  # 覆盖默认的 39，支持更长的 query + instruction
            document_length=2048,  # 支持长文档（测试集最大约2000 tokens）
        )
        
        # 关键：设置 Transformer 的 max_seq_length 和 tokenizer 的 model_max_length
        # 否则训练时会出现截断警告
        max_seq_length = max(512, 2048) + 10  # 加10作为余量
        self.base_model._first_module().max_seq_length = max_seq_length
        # 同时设置 tokenizer 的 model_max_length，这是警告的真正来源
        self.base_model.tokenizer.model_max_length = max_seq_length
        print(f"   设置 max_seq_length = {max_seq_length} (避免截断警告)")
        
        # 重置 IGP 模块状态，确保重新初始化
        self.probe = None
        self.adapter = None
        self.gate = None
        
        # 初始化 IGP 模块
        self._init_igp_modules_from_base_model()
    
    def _init_igp_modules_from_base_model(self):
        """从基础模型初始化 IGP 模块
        
        注意: IGP 模块在 768 维空间操作 (underlying_hidden_size)，
        而不是在 128 维的投影后空间 (actual_embedding_size)。
        这是因为 IGP 需要在投影前注入指令信息。
        """
        underlying_hidden_size = self.base_model[0].get_word_embedding_dimension()
        
        # ColBERT 结构: [0]=LM, [1]=Linear(投影到embedding_size), [-1]=Pool
        # 需要用 [1] 获取 Linear 层的 out_features
        if hasattr(self.base_model[1], 'out_features'):
            actual_embedding_size = self.base_model[1].out_features
        else:
            actual_embedding_size = underlying_hidden_size
        
        print(f"[DEBUG] IGP 模块初始化:")
        print(f"   底层编码器 hidden_size: {underlying_hidden_size}")
        print(f"   实际输出 embedding_size: {actual_embedding_size}")
        print(f"   ⚠️ IGP 模块使用 underlying_hidden_size ({underlying_hidden_size}) 进行初始化")
        
        # IGP 模块在 768 维空间操作
        igp_hidden_size = underlying_hidden_size
        
        if self.enable_probe and self.probe is None:
            self.probe = InstructionProbe(
                hidden_size=igp_hidden_size,
                num_heads=8,
                num_layers=self.probe_num_layers,
                dropout=0.1,
            )
            print(f"✅ InstructionProbe 初始化完成 (hidden_size={igp_hidden_size}, num_layers={self.probe_num_layers})")
        
        if self.enable_adapter and self.adapter is None:
            self.adapter = IGPAdapter(
                hidden_size=igp_hidden_size,
                bottleneck_dim=self.bottleneck_dim,
                dropout=0.1,
                input_dim=igp_hidden_size * 2,  # 拼接 Query 和 Inst_vec
            )
            print(f"✅ IGPAdapter 初始化完成 (bottleneck_dim={self.bottleneck_dim}, input_dim={igp_hidden_size * 2})")
        
        if self.enable_gate and self.gate is None:
            # 默认使用 RatioGateV3（动态感知门控，带L1稀疏正则）
            self.gate = RatioGateV3(
                hidden_size=igp_hidden_size,
                max_ratio=self.max_ratio,
            )
            print(f"✅ RatioGateV3 初始化完成 (max_ratio={self.max_ratio}, 动态感知门控)")
    
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
            self.base_model = models.ColBERT(
                checkpoint_path,
                device=self.device,
                query_length=512,  # 覆盖默认的 39，支持更长的 query + instruction
                document_length=2048,  # 支持长文档（测试集最大约2000 tokens）
            )
            # 关键：设置 Transformer 的 max_seq_length 和 tokenizer 的 model_max_length
            max_seq_length = max(512, 2048) + 10  # 加10作为余量
            self.base_model._first_module().max_seq_length = max_seq_length
            # 同时设置 tokenizer 的 model_max_length，这是警告的真正来源
            self.base_model.tokenizer.model_max_length = max_seq_length
            print(f"✅ 检查点模型已加载到 {self.device}")
            print(f"   设置 max_seq_length = {max_seq_length} (避免截断警告)")
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
        igp_loaded = False
        if os.path.exists(igp_info_path):
            try:
                with open(igp_info_path, 'r') as f:
                    igp_info = json.load(f)
                    print(f"   IGP阶段: {igp_info.get('phase', 'unknown')}")
                    
                    # 初始化 IGP 模块并加载参数
                    self._load_igp_modules(checkpoint_path, igp_info)
                    igp_loaded = True
            except Exception as e:
                print(f"   ⚠️ IGP模块加载跳过: {str(e)}")
        
        # 如果没有 igp_info.json，但启用了 IGP 模块，则初始化新的模块
        if not igp_loaded:
            self._init_igp_modules_from_base_model()
        
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
                        input_dim=hidden_size,
                    )
                self.adapter.load_state_dict(torch.load(adapter_path, map_location='cpu'))
                print(f"   ✅ Adapter 参数已加载")
        
        if 'gate' in modules and self.enable_gate:
            gate_path = os.path.join(checkpoint_path, modules['gate'])
            if os.path.exists(gate_path):
                if self.gate is None:
                    # 尝试根据state_dict判断门控类型
                    gate_state_dict = torch.load(gate_path, map_location='cpu')
                    if any(k.startswith('gate_mlp.') for k in gate_state_dict.keys()):
                        # RatioGateV3 (动态门控)
                        from pylate.models.igp import RatioGateV3
                        self.gate = RatioGateV3(
                            hidden_size=hidden_size,
                            max_ratio=self.max_ratio,
                        )
                        print(f"   📋 检测到 RatioGateV3 格式")
                    elif any(k.startswith('ratio_gate') for k in gate_state_dict.keys()):
                        # RatioGate (静态门控)
                        from pylate.models.igp import RatioGate
                        self.gate = RatioGate(
                            hidden_size=hidden_size,
                            max_ratio=self.max_ratio,
                            use_dynamic=False,
                        )
                        print(f"   📋 检测到 RatioGate 格式")
                    else:
                        # 默认使用 RatioGateV3
                        from pylate.models.igp import RatioGateV3
                        self.gate = RatioGateV3(
                            hidden_size=hidden_size,
                            max_ratio=self.max_ratio,
                        )
                        print(f"   📋 使用默认 RatioGateV3")
                    self.gate.load_state_dict(gate_state_dict)
                else:
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
        if self.base_model is None or self.probe is None:
            return
        
        print("[DEBUG] freeze_all_except_probe: 不冻结参数，只使用极小学习率")
        
        # 不冻结参数！只确保 probe 参数可训练
        # 冻结/解冻会在 optimizer 中通过学习率控制
        
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = False
        
        if self.gate is not None:
            for param in self.gate.parameters():
                param.requires_grad = False
        
        # 确保 probe 参数可训练
        if self.probe is not None:
            for param in self.probe.parameters():
                param.requires_grad = True
        
        # 验证结果
        trainable_count = sum(1 for p in self.probe.parameters() if p.requires_grad)
        frozen_count = sum(1 for p in self.probe.parameters() if not p.requires_grad)
        print(f"[DEBUG] freeze_all_except_probe: probe 可训练={trainable_count}, 冻结={frozen_count}")
    
    def freeze_base_model(self):
        """冻结 base_model 参数（用于 Phase 1），保持 probe 可训练"""
        # 冻结基础模型所有参数
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
        
        # 保持 probe 可训练（不修改其 requires_grad）
        print("🔒 Phase 1: 已冻结 base_model, adapter, gate，保持 probe 可训练")
    
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
        
        # ========== 创建 IGPColBERTWrapper 作为训练模型 ==========
        # Phase 1: 只训练 Probe，冻结其他所有参数
        igp_model = IGPColBERTWrapper(
            base_model=self.base_model,
            probe=self.probe,
            adapter=None,  # Phase 1 不使用 Adapter
            gate=None,     # Phase 1 不使用 Gate
        )
        igp_model.set_phase1_mode()  # 设置 Phase 1 的梯度状态
        
        # 关键：再次确保 base_model 的 max_seq_length 和 tokenizer 的 model_max_length 设置正确
        max_seq_length = max(512, 2048) + 10
        self.base_model._first_module().max_seq_length = max_seq_length
        self.base_model.tokenizer.model_max_length = max_seq_length
        print(f"   [Phase 1] 确认 max_seq_length = {max_seq_length}")
        
        # 配置训练参数
        training_args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(self.output_dir, "phase1"),
            num_train_epochs=self.phase1_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            learning_rate=self.base_lr,  # 这只是默认学习率，我们会用参数分组覆盖
            fp16=False,
            bf16=False,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",  # 禁用自动保存，使用自定义的 EpochCheckpointCallback
            load_best_model_at_end=False,  # 禁用自动加载最佳模型
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.1,
            dataloader_num_workers=4,
        )

        # Phase 1: 只使用 Aux Loss 训练 Probe
        from igp_losses import IGPLossPhase1
        
        igp_loss = IGPLossPhase1(
            base_model=igp_model,  # 传入 IGPColBERTWrapper
            probe=self.probe,
            aux_loss_weight=self.aux_loss_weight,
        )
        
        # 使用参数分组学习率：只优化 Probe 参数
        param_groups = []
        
        # Probe 参数 (使用大学习率)
        if self.probe is not None:
            probe_params = [p for p in self.probe.parameters() if p.requires_grad]
            if probe_params:
                param_groups.append({
                    'params': probe_params,
                    'lr': self.base_lr * 10,  # 显著提高学习率
                })
                print(f"[Phase 1] Probe 参数: {len(probe_params)} 个, lr={self.base_lr * 10}")
        
        if not param_groups:
            raise ValueError("Phase 1: 没有可训练的参数！")
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # DEBUG: 打印参数组信息
        print(f"\n[DEBUG] Phase 1 参数组:")
        for i, group in enumerate(param_groups):
            params_count = len(group['params'])
            grad_count = sum(1 for p in group['params'] if p.requires_grad)
            print(f"   Group {i}: lr={group['lr']}, params={params_count}, requires_grad={grad_count}")
            if params_count > 0:
                total_params = sum(p.numel() for p in group['params'])
                print(f"      Total params: {total_params:,}")
        
        # DEBUG: 打印优化器中的参数
        print(f"\n[DEBUG] 优化器参数状态:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"   Group {i}: {len(group['params'])} params, lr={group['lr']}")
        
        # 创建 Phase 1 输出目录
        phase1_output_dir = os.path.join(self.output_dir, "phase1")
        
        # 创建回调
        early_stopping_callback = EarlyStoppingCallback(
            eval_steps=self.eval_steps,
            patience=3,
            threshold=0.25,
        )
        
        loss_tracker_p1 = LossTracker(phase1_output_dir, None, phase_name="phase1", log_interval=self.log_interval)
        igp_modules_p1 = (self.probe, self.adapter, self.gate)
        best_model_callback_p1 = BestModelCallback(phase1_output_dir, self.base_model, phase_name="phase1", igp_modules=igp_modules_p1)
        checkpoint_callback_p1 = EpochCheckpointCallback(phase1_output_dir, self.base_model, phase_name="phase1", igp_modules=igp_modules_p1)
        
        # 创建训练器
        # 注意：传入 igp_model 作为模型，它会调用 Probe 提取指令向量
        # 通过 optimizers 参数传递自定义优化器，避免被覆盖
        # 使用 IGPColBERTTrainer 确保验证时也使用 IGP 损失
        # 创建 Data Collator
        data_collator = IGPColBERTCollator(
            tokenizer=self.base_model.tokenizer,
            max_query_length=512,
            max_doc_length=2048,  # 支持长文档
        )

        trainer = IGPColBERTTrainer(
            model=igp_model,  # 使用 IGPColBERTWrapper 作为模型
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=igp_loss,  # 传入自定义损失函数
            data_collator=data_collator,
            callbacks=[early_stopping_callback, loss_tracker_p1, best_model_callback_p1, checkpoint_callback_p1],
            optimizers=(optimizer, None),  # 传递自定义优化器，scheduler 为 None
        )
        
        print("📊 开始 Phase 1 训练...")
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
            fp16=False,
            bf16=False,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",  # 禁用自动保存，使用自定义的 EpochCheckpointCallback
            load_best_model_at_end=False,  # 禁用自动加载最佳模型，避免覆盖训练后的参数
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_ratio=0.1,
            dataloader_num_workers=4,
            gradient_accumulation_steps=getattr(self, 'gradient_accumulation_steps', 1),
        )
        
        # ========== 创建 IGPColBERTWrapper 作为训练模型 ==========
        # Phase 1: 只训练 Probe，冻结其他所有参数
        igp_model = IGPColBERTWrapper(
            base_model=self.base_model,
            probe=self.probe,
            adapter=self.adapter,
            gate=self.gate,
        )
        igp_model.set_phase2_mode()  # 设置 Phase 2 的梯度状态
        
        # 关键：再次确保 base_model 的 max_seq_length 和 tokenizer 的 model_max_length 设置正确
        # 因为 IGPColBERTWrapper 可能会影响 base_model 的状态
        max_seq_length = max(512, 2048) + 10
        self.base_model._first_module().max_seq_length = max_seq_length
        self.base_model.tokenizer.model_max_length = max_seq_length
        print(f"   [Phase 2] 确认 max_seq_length = {max_seq_length}")
        
        # 基础损失
        base_loss = losses.Contrastive(model=self.base_model)
        
        # IGP 损失 (只负责计算损失，IGP 模块调用在 Model.forward 中)
        igp_loss = IGPLoss(
            base_loss=base_loss,
            base_model=igp_model,  # 传入 IGPColBERTWrapper
            probe=self.probe,
            adapter=self.adapter,
            gate=self.gate,
            aux_loss_weight=self.aux_loss_weight,
            gate_l1_coeff=self.gate_l1_coeff,  # 兼容旧参数，实际不再使用
            lambda_gate=self.lambda_gate,  # 门控监督损失权重
        )
        
        # 创建优化器，为门控分配独立学习率
        param_groups = []
        
        # 基础模型参数 (冻结，不加入优化器)
        # Probe 参数 (使用基础学习率)
        if self.probe is not None:
            probe_params = [p for p in self.probe.parameters() if p.requires_grad]
            if probe_params:
                param_groups.append({
                    'params': probe_params,
                    'lr': self.base_lr,
                })
                print(f"[Phase 2] Probe 参数: {len(probe_params)} 个, lr={self.base_lr}")
        
        # Adapter 参数 (使用基础学习率)
        if self.adapter is not None:
            adapter_params = [p for p in self.adapter.parameters() if p.requires_grad]
            if adapter_params:
                param_groups.append({
                    'params': adapter_params,
                    'lr': self.base_lr,
                })
                print(f"[Phase 2] Adapter 参数: {len(adapter_params)} 个, lr={self.base_lr}")
        
        # Gate 参数 (使用大学习率)
        if self.gate is not None:
            gate_params = [p for p in self.gate.parameters() if p.requires_grad]
            if gate_params:
                param_groups.append({
                    'params': gate_params,
                    'lr': self.gate_lr,  # 大学习率，确保门控能被激活
                })
                print(f"[Phase 2] Gate 参数: {len(gate_params)} 个, lr={self.gate_lr}")
        
        if not param_groups:
            raise ValueError("Phase 2: 没有可训练的参数！")
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # 创建学习率调度器 (warmup + cosine decay)
        from transformers import get_cosine_schedule_with_warmup
        num_training_steps = len(self.train_dataset) // self.batch_size * self.phase2_epochs
        num_warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print(f"[Phase 2] 学习率调度器: warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
        
        # 创建 Phase 2 输出目录
        phase2_output_dir = os.path.join(self.output_dir, "phase2")
        
        # 创建回调
        loss_tracker_p2 = LossTracker(phase2_output_dir, None, phase_name="phase2", log_interval=self.log_interval)
        igp_modules_p2 = (self.probe, self.adapter, self.gate)
        best_model_callback_p2 = BestModelCallback(phase2_output_dir, self.base_model, phase_name="phase2", igp_modules=igp_modules_p2)
        checkpoint_callback_p2 = EpochCheckpointCallback(phase2_output_dir, self.base_model, phase_name="phase2", igp_modules=igp_modules_p2)
        
        phase2_early_stopping = Phase2EarlyStoppingCallback(
            patience=self.phase2_patience,
            threshold=self.phase2_early_stop_threshold,
            mode='min',
        )
        
        # 创建 Data Collator
        data_collator = IGPColBERTCollator(
            tokenizer=self.base_model.tokenizer,
            max_query_length=512,
            max_doc_length=2048,  # 支持长文档
        )

        # 创建训练器
        # 注意：传入 igp_model 作为模型，它会调用 IGP 模块
        # 通过 optimizers 参数传递自定义优化器，避免被覆盖
        # 使用 IGPColBERTTrainer 确保验证时也使用 IGP 损失
        trainer = IGPColBERTTrainer(
            model=igp_model,  # 使用 IGPColBERTWrapper 作为模型
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            loss=igp_loss,  # 传入自定义损失函数
            data_collator=data_collator,
            callbacks=[loss_tracker_p2, best_model_callback_p2, phase2_early_stopping, checkpoint_callback_p2],
            optimizers=(optimizer, scheduler),  # 传递自定义优化器和调度器
        )
        
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
        
        print(f"\n[DEBUG] train_phase2 结束 - probe: {self.probe}, adapter: {self.adapter}, gate: {self.gate}")
        
        return phase2_best_loss
    
    def train(self):
        """执行完整的两阶段训练"""
        print("=" * 60)
        print("📊 ColBERT-IGP 两阶段训练")
        print("=" * 60)
        
        # Phase 2 独立训练模式验证
        if not self.enable_phase1 and self.enable_phase2:
            print("\n⚠️ Phase 2 独立训练模式 (跳过 Phase 1)")
            if self.phase1_checkpoint:
                print(f"✅ Phase 1 检查点已指定: {self.phase1_checkpoint}")
            else:
                print("   将使用随机初始化的 IGP 模块")
        
        # 当启用 Phase2 且跳过 Phase1 时，需要先加载模型
        if self.enable_phase2 and not self.enable_phase1 and self.phase1_checkpoint:
            print("\n📂 预加载 Phase 1 检查点...")
            self.load_checkpoint(self.phase1_checkpoint, phase="phase1")
        
        # 加载数据
        self.load_model()  # 先加载模型
        self.load_data()  # 再加载数据
        
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
                    print(f"✅ 检查点加载完成，将继续训练 Phase 2")
                except Exception as e:
                    print(f"❌ 加载 Phase 2 检查点失败: {str(e)}")
                    raise
                
                print("\n⏩ Phase 2 从检查点继续训练")
                phase2_best_loss = self.train_phase2()
            else:
                # 检查是否需要从 Phase 1 检查点加载
                # 如果已经预加载过了，就跳过
                if (not self.enable_phase1 or self.phase1_checkpoint) and self.base_model is None:
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
            probe_sd = probe.state_dict()
            if any(k.startswith('igp_probe.') for k in probe_sd.keys()):
                new_sd = {k.replace('igp_probe.', ''): v for k, v in probe_sd.items()}
                torch.save(new_sd, probe_path)
            else:
                torch.save(probe_sd, probe_path)
            igp_state['probe'] = 'igp_probe.pt'
            print(f"   ✅ Probe 参数已保存: {probe_path}")
        
        if adapter is not None and hasattr(adapter, 'state_dict'):
            adapter_path = os.path.join(save_dir, "igp_adapter.pt")
            adapter_sd = adapter.state_dict()
            if any(k.startswith('igp_adapter.') for k in adapter_sd.keys()):
                new_sd = {k.replace('igp_adapter.', ''): v for k, v in adapter_sd.items()}
                torch.save(new_sd, adapter_path)
            else:
                torch.save(adapter_sd, adapter_path)
            igp_state['adapter'] = 'igp_adapter.pt'
            print(f"   ✅ Adapter 参数已保存: {adapter_path}")
        
        if gate is not None and hasattr(gate, 'state_dict'):
            gate_path = os.path.join(save_dir, "igp_gate.pt")
            gate_sd = gate.state_dict()
            if any(k.startswith('igp_gate.') for k in gate_sd.keys()):
                new_sd = {k.replace('igp_gate.', ''): v for k, v in gate_sd.items()}
                torch.save(new_sd, gate_path)
            else:
                torch.save(gate_sd, gate_path)
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
        else:
            print(f"   ⚠️ 没有可保存的 IGP 模块参数")
    
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
                            'anchor': query,
                            'positive': pos_doc,
                            'negative': neg_doc,
                        })
    
    print(f"✅ 加载了 {len(data_list)} 个训练样本")
    dataset = Dataset.from_list(data_list)
    # 确保列顺序符合预期: anchor, positive, negative
    dataset = dataset.select_columns(['anchor', 'positive', 'negative'])
    return dataset


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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数 (等效batch size = batch_size * gradient_accumulation_steps)')
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
    parser.add_argument('--probe_num_layers', type=int, default=2,
                        help='InstructionProbe 编码器层数')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='辅助损失权重')
    parser.add_argument('--gate_l1_coeff', type=float, default=0.01,
                        help='兼容旧参数，实际不再使用')
    parser.add_argument('--lambda_gate', type=float, default=1.0,
                        help='门控监督损失权重 (默认1.0，建议范围0.5-2.0)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='损失记录间隔 (每多少个step记录一次，用于生成更细致的损失曲线，设为0则记录每个step)')
    
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
    
    # 设置设备 - 直接使用用户指定的设备，不覆盖环境变量
    
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
        probe_num_layers=args.probe_num_layers,
        aux_loss_weight=args.aux_loss_weight,
        gate_l1_coeff=args.gate_l1_coeff,  # 兼容旧参数，实际不再使用
        lambda_gate=args.lambda_gate,  # 门控监督损失权重
        phase2_patience=args.phase2_patience,
        phase2_early_stop_threshold=args.phase2_early_stop_threshold,
        save_total_limit=args.save_total_limit,
        phase1_checkpoint=args.phase1_checkpoint,
        phase2_checkpoint=args.phase2_checkpoint,
        log_interval=args.log_interval,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    trainer.train()


if __name__ == '__main__':
    main()
