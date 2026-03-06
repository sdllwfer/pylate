"""
ColBERT 在 FollowIR 训练数据上进行微调
"""

import os
import sys

# 激活 conda 环境
conda_env = os.environ.get('CONDA_DEFAULT_ENV')
if conda_env != 'pylate':
    import subprocess
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    conda_path = subprocess.run(['which', 'conda'], capture_output=True, text=True).stdout.strip()
    if conda_path:
        env_python = f"{os.path.dirname(conda_path)}/envs/pylate/bin/python"
        if os.path.exists(env_python):
            os.execv(env_python, [sys.executable] + sys.argv)

import argparse
import json
import os
import sys

# 强制刷新输出
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

from datetime import datetime

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import TrainerCallback

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ matplotlib 不可用: {e}")
    MATPLOTLIB_AVAILABLE = False

from pylate import losses, models, utils


class LossRecorderCallback(TrainerCallback):
    """自定义回调：记录每个epoch的训练和验证损失"""
    
    def __init__(self, output_dir, trainer):
        self.output_dir = output_dir
        self.trainer = trainer
        self.train_losses = []
        self.eval_losses = []
        self.epochs = []
        self.current_train_loss = None
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            self.current_train_loss = logs.get('loss')
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics.get('eval_loss')
            epoch = int(state.epoch)
            
            while len(self.eval_losses) < epoch:
                self.eval_losses.append(None)
            
            if epoch > 0:
                self.eval_losses[epoch - 1] = eval_loss
            
            self._save_csv()
            self._plot_losses()
            print(f"\n📊 验证损失已记录: epoch={epoch}, eval_loss={eval_loss:.6f}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = kwargs.get('metrics', {})
        
        epoch = int(state.epoch)
        
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            if self.current_train_loss is not None:
                self.train_losses.append(self.current_train_loss)
            self.current_train_loss = None
        
        self._save_csv()
        self._plot_losses()
    
    def _save_csv(self):
        csv_path = os.path.join(self.output_dir, "loss_history.csv")
        with open(csv_path, 'w') as f:
            f.write("epoch,train_loss,eval_loss\n")
            for i in range(len(self.epochs)):
                train_loss = self.train_losses[i] if i < len(self.train_losses) and self.train_losses[i] is not None else ""
                eval_loss = self.eval_losses[i] if i < len(self.eval_losses) and self.eval_losses[i] is not None else ""
                f.write(f"{self.epochs[i]},{train_loss},{eval_loss}\n")
    
    def _plot_losses(self):
        if not MATPLOTLIB_AVAILABLE:
            return
        
        train_epochs = [self.epochs[i] for i in range(len(self.epochs)) if i < len(self.train_losses) and self.train_losses[i] is not None]
        train_losses_filtered = [l for l in self.train_losses if l is not None]
        
        eval_epochs = [self.epochs[i] for i in range(len(self.epochs)) if i < len(self.eval_losses) and self.eval_losses[i] is not None]
        eval_losses_filtered = [l for l in self.eval_losses if l is not None]
        
        if not train_losses_filtered and not eval_losses_filtered:
            return
        
        plt.figure(figsize=(10, 6))
        
        if train_losses_filtered:
            plt.plot(train_epochs, train_losses_filtered, 
                     'b-o', label='Training Loss', linewidth=2, markersize=8)
        if eval_losses_filtered:
            plt.plot(eval_epochs, eval_losses_filtered, 
                     'r-s', label='Validation Loss', linewidth=2, markersize=8)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(self.epochs)
        
        chart_path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 损失曲线已更新: {chart_path}")


class BestModelCallback(TrainerCallback):
    """自定义回调：保存验证集损失最低的模型"""
    
    def __init__(self, output_dir, model, trainer=None):
        self.output_dir = output_dir
        self.model = model
        self.trainer = trainer
        self.best_eval_loss = float('inf')
        self.best_model_path = None
        self.save_dir = os.path.join(output_dir, "best_model")
        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics.get('eval_loss')
            
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                epoch = int(state.epoch)
                
                import shutil
                if os.path.exists(self.save_dir):
                    shutil.rmtree(self.save_dir)
                os.makedirs(self.save_dir, exist_ok=True)
                
                self.model.save(self.save_dir)
                
                self._save_trainer_state(state, epoch, eval_loss)
                
                print(f"\n🏆 保存最佳模型 (epoch={epoch}, 验证损失: {eval_loss:.6f}): {self.save_dir}")
                
                self._save_best_info(epoch, eval_loss)
    
    def _save_trainer_state(self, state, epoch, eval_loss):
        """保存训练器状态，使 best_model 可以作为检查点恢复训练"""
        import json
        import torch
        import numpy as np
        
        trainer_state = {
            "epoch": epoch,
            "global_step": state.global_step,
            "best_metric": self.best_eval_loss,
            "best_model_checkpoint": self.save_dir,
            "is_local_process_zero": True,
            "is_world_process_zero": True,
        }
        
        with open(os.path.join(self.save_dir, "trainer_state.json"), 'w') as f:
            json.dump(trainer_state, f)
        
        if self.trainer is not None and hasattr(self.trainer, 'args') and self.trainer.args is not None:
            training_args = {
                "num_train_epochs": self.trainer.args.num_train_epochs,
                "per_device_train_batch_size": self.trainer.args.per_device_train_batch_size,
                "per_device_eval_batch_size": self.trainer.args.per_device_eval_batch_size,
                "learning_rate": self.trainer.args.learning_rate,
                "output_dir": self.output_dir,
            }
            
            with open(os.path.join(self.save_dir, "training_args.bin"), 'wb') as f:
                import pickle
                pickle.dump(training_args, f)
        
        if self.trainer is not None and hasattr(self.trainer, 'optimizer'):
            try:
                torch.save(self.trainer.optimizer.state_dict(), 
                          os.path.join(self.save_dir, "optimizer.pt"))
            except Exception as e:
                print(f"  ⚠️ 无法保存优化器状态: {e}")
        
        if self.trainer is not None and hasattr(self.trainer, 'lr_scheduler'):
            try:
                torch.save(self.trainer.lr_scheduler.state_dict(), 
                          os.path.join(self.save_dir, "scheduler.pt"))
            except Exception as e:
                print(f"  ⚠️ 无法保存学习率调度器状态: {e}")
        
        try:
            torch.save(torch.get_rng_state(), os.path.join(self.save_dir, "rng_state.pth"))
            np.save(os.path.join(self.save_dir, "rng_state.npy"), np.random.get_state())
        except Exception as e:
            print(f"  ⚠️ 无法保存随机状态: {e}")
    
    def _save_best_info(self, epoch, eval_loss):
        info_path = os.path.join(self.output_dir, "best_model_info.json")
        info = {
            "epoch": epoch,
            "eval_loss": eval_loss,
            "model_path": self.best_model_path,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)


def load_followir_train_data(data_path):
    """加载 FollowIR 训练数据并转换为训练格式"""
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
    parser = argparse.ArgumentParser(description='ColBERT FollowIR 微调')
    parser.add_argument('--model_name', type=str, default='lightonai/GTE-ModernColBERT-v1',
                        help='基座模型名称或路径')
    parser.add_argument('--train_data', type=str, 
                        default='/home/luwa/Documents/pylate/dataset/colbert_data/FollowIR_train/colbert_train_final.jsonl',
                        help='训练数据路径')
    parser.add_argument('--output_dir', type=str, 
                        default='/home/luwa/Documents/pylate/output/colbert_finetune_followir',
                        help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='从检查点继续训练，传入检查点路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='GPU 设备')
    parser.add_argument('--save_total_limit', type=int, default=10,
                        help='最多保存多少个候选最佳模型')
    parser.add_argument('--note', type=str, default='',
                        help='备注信息，会记录到参数文件中')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("📊 ColBERT FollowIR 微调")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"训练数据: {args.train_data}")
    print(f"输出目录: {args.output_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"GPU: {args.device}")
    print("=" * 60)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]

    train_dataset = load_followir_train_data(args.train_data)

    splits = train_dataset.train_test_split(test_size=0.1)
    train_data = splits['train']
    eval_data = splits['test']

    print(f"\n训练集: {len(train_data)} 样本")
    print(f"验证集: {len(eval_data)} 样本")

    print(f"\n📥 加载模型: {args.model_name}")
    model = models.ColBERT(model_name_or_path=args.model_name)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.1,
        dataloader_num_workers=4,
    )

    train_loss = losses.Contrastive(model=model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        loss=train_loss,
        data_collator=utils.ColBERTCollator(tokenize_fn=model.tokenize),
    )

    loss_callback = LossRecorderCallback(args.output_dir, trainer)
    best_model_callback = BestModelCallback(args.output_dir, model, trainer)
    trainer.add_callback(loss_callback)
    trainer.add_callback(best_model_callback)

    params_file = os.path.join(args.output_dir, "training_params.txt")
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ColBERT FollowIR 训练参数\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"训练数据: {args.train_data}\n")
        f.write(f"输出目录: {args.output_dir}\n")
        f.write(f"GPU设备: {args.device}\n")
        f.write(f"训练轮数: {args.num_epochs}\n")
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"学习率: {args.learning_rate}\n")
        f.write(f"训练样本数: {len(train_data)}\n")
        f.write(f"验证样本数: {len(eval_data)}\n")
        f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if args.resume_from_checkpoint:
            f.write(f"起始检查点: {args.resume_from_checkpoint}\n")
        if args.note:
            f.write(f"备注: {args.note}\n")
        f.write("=" * 60 + "\n")
    print(f"📝 参数已保存至: {params_file}")

    print("\n🚀 开始训练...")
    
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        
        if os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
            try:
                print(f"✅ 从检查点恢复训练: {checkpoint_path}")
                trainer.train(resume_from_checkpoint=checkpoint_path)
            except ValueError as e:
                if "CVE-2025-32434" in str(e) or "torch.load" in str(e):
                    print(f"⚠️ PyTorch 版本过低，无法加载优化器状态，将从头开始训练")
                    trainer.train()
                else:
                    raise
        elif os.path.basename(checkpoint_path) == "best_model" or checkpoint_path.endswith("/best_model"):
            base_dir = os.path.dirname(checkpoint_path)
            checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                checkpoint_path = os.path.join(base_dir, latest_checkpoint)
                print(f"⚠️ best_model 目录不支持恢复训练，尝试最新检查点: {checkpoint_path}")
                try:
                    trainer.train(resume_from_checkpoint=checkpoint_path)
                except ValueError as e:
                    if "CVE-2025-32434" in str(e) or "torch.load" in str(e):
                        print(f"⚠️ PyTorch 版本过低，无法加载优化器状态，将从头开始训练")
                        trainer.train()
                    else:
                        raise
            else:
                print(f"⚠️ 未找到检查点，将从头开始训练")
                trainer.train()
        else:
            try:
                print(f"✅ 从检查点恢复训练: {checkpoint_path}")
                trainer.train(resume_from_checkpoint=checkpoint_path)
            except ValueError as e:
                if "CVE-2025-32434" in str(e) or "torch.load" in str(e):
                    print(f"⚠️ PyTorch 版本过低，无法加载优化器状态，将从头开始训练")
                    trainer.train()
                else:
                    raise
    else:
        trainer.train()

    model_save_path = os.path.join(args.output_dir, "final_model")
    model.save(model_save_path)
    print(f"\n✅ 模型已保存至: {model_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
