#!/usr/bin/env python3
"""
Recovery Training Script
Optimized to recover the lost model's performance (66.54% test, 33% multi-F1)
Uses proven techniques to get accuracy back up
"""

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from combined_preprocessing import create_combined_data_loaders
from lightweight_model import LightweightVQAModel


class ImprovedFocalLoss(nn.Module):
    """Focal Loss with per-class weights."""
    
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class EarlyStopping:
    """Early stopping with more patience."""
    
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        
        return self.early_stop


class RecoveryTrainer:
    """Trainer optimized to match original performance."""
    
    def __init__(self,
                 num_answers: int,
                 batch_size: int = 24,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 30,
                 device: str = None,
                 checkpoint_dir: str = 'checkpoints/recovery_training',
                 # Optimized parameters based on successful training
                 fusion_hidden_dim: int = 384,
                 num_attention_heads: int = 6,
                 dropout: float = 0.35,  # Slightly reduced
                 focal_gamma: float = 2.5,  # Balanced
                 weight_decay: float = 0.007,  # Moderate
                 early_stopping_patience: int = 10):  # More patience
        
        self.num_answers = num_answers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.focal_gamma = focal_gamma
        self.weight_decay = weight_decay
        
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.setup_logging()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.class_weights = None
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
    
    def setup_logging(self):
        log_file = os.path.join(self.checkpoint_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Calculate balanced class weights."""
        self.logger.info("Calculating class weights...")
        all_labels = []
        for batch in tqdm(train_loader, desc='Analyzing class distribution'):
            all_labels.extend(batch['answer_idx'].tolist())
        
        label_counts = Counter(all_labels)
        total = len(all_labels)
        
        weights = torch.zeros(self.num_answers)
        for label in range(self.num_answers):
            count = label_counts.get(label, 0)
            if count > 0:
                # Smoothed inverse frequency
                weights[label] = np.sqrt(total / (count + 50))
            else:
                weights[label] = 0.1
        
        weights = weights / weights.mean()
        weights = torch.clamp(weights, min=0.3, max=5.0)
        
        self.logger.info(f"Class weights - min: {weights.min():.3f}, "
                        f"max: {weights.max():.3f}, mean: {weights.mean():.3f}")
        
        return weights
    
    def initialize_model(self, train_loader: DataLoader):
        """Initialize model."""
        self.logger.info("=" * 80)
        self.logger.info("INITIALIZING RECOVERY TRAINING")
        self.logger.info("=" * 80)
        self.logger.info(f"Target: Match original performance (66.54% test, 33% multi-F1)")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 80)
        
        self.model = LightweightVQAModel(
            num_classes=self.num_answers,
            fusion_hidden_dim=self.fusion_hidden_dim,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            freeze_vision_encoder=False,
            freeze_text_encoder=False,
            use_spatial_features=False
        ).to(self.device)
        
        self.class_weights = self.calculate_class_weights(train_loader).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
            eps=1e-8
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=4,  # More patience
            min_lr=1e-7
        )
        
        self.criterion = ImprovedFocalLoss(
            alpha=self.class_weights,
            gamma=self.focal_gamma
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"✅ Model initialized - {trainable_params:,} trainable params")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch} Training')
        for batch in progress:
            images = batch['image'].to(self.device)
            question_ids = batch['question']['input_ids'].to(self.device)
            question_mask = batch['question']['attention_mask'].to(self.device)
            answers = batch['answer_idx'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(images, question_ids, question_mask)
            loss = self.criterion(logits, answers)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == answers).sum().item()
            total += answers.size(0)
            
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        return total_loss / len(train_loader), 100 * correct / total
    
    def validate(self, val_loader: DataLoader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress = tqdm(val_loader, desc='Validation')
            for batch in progress:
                images = batch['image'].to(self.device)
                question_ids = batch['question']['input_ids'].to(self.device)
                question_mask = batch['question']['attention_mask'].to(self.device)
                answers = batch['answer_idx'].to(self.device)
                
                logits = self.model(images, question_ids, question_mask)
                loss = self.criterion(logits, answers)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == answers).sum().item()
                total += answers.size(0)
                
                progress.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        return total_loss / len(val_loader), 100 * correct / total
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'history': self.history,
            'config': {
                'num_answers': self.num_answers,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'architecture': 'lightweight_attention_recovery',
                'fusion_hidden_dim': self.fusion_hidden_dim,
                'num_attention_heads': self.num_attention_heads,
                'dropout': self.dropout,
                'focal_gamma': self.focal_gamma,
                'weight_decay': self.weight_decay,
            }
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model_recovered.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🌟 Saved BEST model (Val Acc: {val_acc:.2f}%)")
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
    def plot_training_history(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # ---- (1) LOSS CURVE ----
        axes[0, 0].plot(epochs, self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # ---- (2) ACCURACY CURVE ----
        axes[0, 1].plot(epochs, self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # ---- (3) LEARNING RATE ----
        axes[1, 0].plot(epochs, self.history['learning_rates'], color='green', linewidth=2)
        axes[1, 0].set_title('Learning Rate (log scale)')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # ---- (4) OVERFITTING GAP ----
        gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1, 1].plot(epochs, gap, color='purple', linewidth=2)
        axes[1, 1].set_title('Overfitting Gap (Train - Val)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(self.checkpoint_dir, 'training_curves_simple.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"📊 Training curves saved to: {plot_path}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        start_time = datetime.now()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RECOVERY TRAINING START")
        self.logger.info("=" * 80)
        
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch}/{self.num_epochs}")
            self.logger.info(f"{'='*80}")
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.logger.info(f"✓ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            val_loss, val_acc = self.validate(val_loader)
            self.logger.info(f"✓ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Check if we've matched or exceeded original performance
            if val_acc >= 64.0 and train_acc >= 70.0:
                self.logger.info(f"🎯 Good performance achieved! Val: {val_acc:.2f}%, Train: {train_acc:.2f}%")
            
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"📉 Learning rate: {current_lr:.2e}")
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
            else:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
            
            if self.early_stopping(val_acc):
                self.logger.info(f"\n⚠️  Early stopping at epoch {epoch}")
                break
        
        duration = (datetime.now() - start_time).total_seconds()
        
        self.plot_training_history()
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🎉 RECOVERY TRAINING COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"Time: {duration/60:.2f} minutes")
        self.logger.info(f"Best Val Acc: {self.best_val_acc:.2f}%")
        self.logger.info(f"Target was: 63.94% val, 66.54% test, 33% multi-F1")
        self.logger.info("=" * 80)


def main():
    params = {
        'batch_size': 24,
        'learning_rate': 1e-4,
        'num_epochs': 30,
        'max_answer_vocab_size': 250,
        'fusion_hidden_dim': 384,
        'num_attention_heads': 6,
        'dropout': 0.35,  # Slightly reduced from 0.4
        'focal_gamma': 2.5,  # Balanced
        'weight_decay': 0.007,  # Moderate
        'early_stopping_patience': 10  # More patience
    }
    
    print("=" * 80)
    print("🔧 RECOVERY TRAINING")
    print("Target: Match original performance (66.54% test, 33% multi-F1)")
    print("=" * 80)
    print("\n📋 Optimized Configuration:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    print("=" * 80)
    
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=params['batch_size'],
        max_samples=None,
        num_workers=0,
        max_answer_vocab_size=params['max_answer_vocab_size']
    )
    
    num_answers = len(answer_vocab)
    print(f"✓ Answers: {num_answers}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    trainer = RecoveryTrainer(
        num_answers=num_answers,
        **{k: v for k, v in params.items() if k not in ['max_answer_vocab_size']}
    )
    
    trainer.initialize_model(train_loader)
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training complete! Evaluate with:")
    print("  python evaluate_with_roc.py --checkpoint checkpoints/recovery_training/best_model_recovered.pt")


if __name__ == "__main__":
    main()

