#!/usr/bin/env python3
"""
Training script for baseline attention-fusion model (ResNet34 + BERT).
Includes class weights, focal loss, cosine schedule, and logging.
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from baseline_model_attention import BaselineVQAModel, print_model_summary
from combined_preprocessing import create_combined_data_loaders
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class UltimateVQATrainer:
    def __init__(self,
                 num_answers: int,
                 batch_size: int = 24,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 15,
                 device: str = None,
                 checkpoint_dir: str = 'checkpoints/baseline_attention',
                 use_focal_loss: bool = True,
                 use_class_weights: bool = False):
        """
        Initialize ultimate trainer with all improvements.
        """
        self.num_answers = num_answers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_focal_loss = use_focal_loss
        self.use_class_weights = use_class_weights
        
        # Device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps') 
        else:
            self.device = torch.device('cpu')
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.setup_logging()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.class_weights = None
        
        # Training history
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
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('training_baseline_attention.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_class_weights(self, train_loader):
        """Calculate class weights for imbalanced data."""
        self.logger.info("Calculating class weights...")
        
        all_labels = []
        for batch in tqdm(train_loader, desc='Analyzing class distribution'):
            all_labels.extend(batch['answer_idx'].tolist())
        
        label_counts = Counter(all_labels)
        total = len(all_labels)
        
        self.logger.info(f"Total samples: {total}")
        self.logger.info(f"Number of classes: {len(label_counts)}")
        self.logger.info(f"Most common: {label_counts.most_common(5)}")
        self.logger.info(f"Least common: {list(label_counts.most_common())[-5:]}")
        
        # Inverse frequency weighting with smoothing
        weights = torch.zeros(self.num_answers)
        for label in range(self.num_answers):
            count = label_counts.get(label, 0)
            if count > 0:
                # Smooth inverse frequency
                weights[label] = np.log(total / (count + 10))
            else:
                weights[label] = 0.0
        
        # Normalize weights
        weights = weights / weights.sum() * self.num_answers
        weights = torch.clamp(weights, min=0.1, max=10.0)
        
        self.logger.info(f"Class weights - min: {weights.min():.3f}, max: {weights.max():.3f}, mean: {weights.mean():.3f}")
        
        return weights
    
    def create_model(self):
        """Create the attention-fusion baseline model (ResNet34 + BERT)."""
        model = BaselineVQAModel(
            num_classes=self.num_answers,
            visual_feature_dim=512,
            text_feature_dim=768,
            fusion_hidden_dim=512,
            num_attention_heads=4,
            dropout=0.3,
            freeze_vision_encoder=False,
            freeze_text_encoder=False
        )
        return model
    
    def initialize_model(self, train_loader):
        """Initialize model with all improvements."""
        self.logger.info(f"Initializing model on device: {self.device}")
        
        # Create attention-fusion baseline model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        print_model_summary(self.model)
        
        # Calculate class weights
        if self.use_class_weights:
            self.class_weights = self.calculate_class_weights(train_loader)
            self.class_weights = self.class_weights.to(self.device)
        
        # Better optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # Restart every 5 epochs
            T_mult=2,
            eta_min=1e-7
        )
        
        # Loss function
        if self.use_focal_loss:
            self.logger.info("Using Focal Loss")
            self.criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif self.use_class_weights:
            self.logger.info("Using Weighted CrossEntropy")
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                label_smoothing=0.1
            )
        else:
            self.logger.info("Using Standard CrossEntropy")
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.logger.info(f"✅ Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            question_ids = batch['question']['input_ids'].to(self.device)
            question_mask = batch['question']['attention_mask'].to(self.device)
            answers = batch['answer_idx'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images, question_ids, question_mask)
            loss = self.criterion(logits, answers)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == answers).sum().item()
            total += answers.size(0)
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='Validation')
            for batch in progress_bar:
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
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        val_loss = total_loss / len(val_loader)
        val_acc = 100 * correct / total
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
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
                'answer_vocab_size': self.num_answers,
                'use_focal_loss': self.use_focal_loss,
                'use_class_weights': self.use_class_weights
            }
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🌟 Saved BEST model (Val Acc: {val_acc:.2f}%)")
    
    def plot_training_history(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gap between train and val
        gap = [t - v for t, v in zip(self.history['train_acc'], self.history['val_acc'])]
        axes[1, 1].plot(epochs, gap, 'purple', linewidth=2)
        axes[1, 1].set_title('Overfitting Gap (Train - Val Acc)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Gap (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.checkpoint_dir, 'training_curves_baseline.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Plots saved to {plot_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model."""
        start_time = datetime.now()
        
        self.logger.info("=" * 80)
        self.logger.info("BASELINE TRAINING")
        self.logger.info("=" * 80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.num_epochs}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Focal Loss: {self.use_focal_loss}")
        self.logger.info(f"Class Weights: {self.use_class_weights}")
        self.logger.info("=" * 80)
        
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch}/{self.num_epochs}")
            self.logger.info(f"{'='*80}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.logger.info(f"✓ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.logger.info(f"✓ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"📉 Learning rate: {current_lr:.2e}")
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=True)
        
        # Training completed
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {duration/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.plot_training_history()
        
        # Save final model
        final_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'num_answers': self.num_answers,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'answer_vocab_size': self.num_answers
            }
        }
        final_path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        torch.save(final_checkpoint, final_path)
        self.logger.info(f"💾 Final model saved to {final_path}")


def main():
    """Main training with all improvements."""
    
    params = {
        'batch_size': 24,
        'learning_rate': 1e-4,
        'num_epochs': 15,
        'max_samples': None,
        'max_answer_vocab_size': 120,  # Further reduced
        'use_focal_loss': True,
        'use_class_weights': False,  # Don't use both together
    }
    
    print("=" * 80)
    print("BASELINE TRAINING CONFIGURATION")
    print("=" * 80)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=params['batch_size'],
        max_samples=params['max_samples'],
        num_workers=0,
        max_answer_vocab_size=params['max_answer_vocab_size']
    )
    
    num_answers = len(answer_vocab)
    print(f"✓ Vocabulary: {num_answers} answers")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = UltimateVQATrainer(
        num_answers=num_answers,
        batch_size=params['batch_size'],
        learning_rate=params['learning_rate'],
        num_epochs=params['num_epochs'],
        use_focal_loss=params['use_focal_loss'],
        use_class_weights=params['use_class_weights']
    )
    
    # Initialize model
    trainer.initialize_model(train_loader)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print("\nRun: python evaluate_saved_model.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
