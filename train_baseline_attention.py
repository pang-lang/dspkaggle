#!/usr/bin/env python3
"""
Training script for baseline attention-fusion model (ResNet34 + BERT).
MATCHED TO LIGHTWEIGHT PARAMETERS for fair comparison.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau  # CHANGED: Match lightweight
from models.baseline_model_attention import BaselineVQAModel, print_model_summary
from preprocessing.combined_preprocessing import create_combined_data_loaders
import numpy as np
from tqdm import tqdm
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter


class ImprovedFocalLoss(nn.Module):
    """Focal Loss with per-class weights - MATCHED to lightweight."""
    
    def __init__(self, alpha=None, gamma=2.5, reduction='mean'):  # CHANGED: gamma=2.5
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
    """Early stopping - MATCHED to lightweight."""
    
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


class MatchedBaselineTrainer:
    def __init__(self,
                 num_answers: int,
                 batch_size: int = 24,
                 learning_rate: float = 1e-4,
                 num_epochs: int = 30,  # CHANGED: Match lightweight
                 device: str = None,
                 checkpoint_dir: str = 'checkpoints/baseline_matched',
                 # MATCHED PARAMETERS
                 fusion_hidden_dim: int = 384,  # CHANGED: Was 512
                 num_attention_heads: int = 6,  # CHANGED: Was 4
                 dropout: float = 0.35,  # CHANGED: Was 0.3
                 focal_gamma: float = 2.5,  # CHANGED: Was 2.0
                 weight_decay: float = 0.007,  # CHANGED: Was 0.01
                 early_stopping_patience: int = 10):  # NEW
        """
        Initialize trainer with parameters MATCHED to lightweight model.
        """
        self.num_answers = num_answers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        
        # Architecture parameters
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.focal_gamma = focal_gamma
        self.weight_decay = weight_decay
        
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
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        
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
    
    def calculate_class_weights(self, train_loader):
        """Calculate class weights - MATCHED to lightweight."""
        self.logger.info("Calculating class weights...")
        
        all_labels = []
        for batch in tqdm(train_loader, desc='Analyzing class distribution'):
            all_labels.extend(batch['answer_idx'].tolist())
        
        label_counts = Counter(all_labels)
        total = len(all_labels)
        
        self.logger.info(f"Total samples: {total}")
        self.logger.info(f"Number of classes: {len(label_counts)}")
        
        # MATCHED weighting scheme from lightweight
        weights = torch.zeros(self.num_answers)
        for label in range(self.num_answers):
            count = label_counts.get(label, 0)
            if count > 0:
                # Smoothed inverse frequency with sqrt
                weights[label] = np.sqrt(total / (count + 50))
            else:
                weights[label] = 0.1
        
        weights = weights / weights.mean()
        weights = torch.clamp(weights, min=0.3, max=5.0)
        
        self.logger.info(f"Class weights - min: {weights.min():.3f}, max: {weights.max():.3f}, mean: {weights.mean():.3f}")
        
        return weights
    
    def create_model(self):
        """Create the attention-fusion baseline model with MATCHED parameters."""
        model = BaselineVQAModel(
            num_classes=self.num_answers,
            visual_feature_dim=512,
            text_feature_dim=768,
            fusion_hidden_dim=self.fusion_hidden_dim,  # Now 384
            num_attention_heads=self.num_attention_heads,  # Now 6
            dropout=self.dropout,  # Now 0.35
            freeze_vision_encoder=False,
            freeze_text_encoder=False
        )
        return model
    
    def initialize_model(self, train_loader):
        """Initialize model with MATCHED improvements."""
        self.logger.info("=" * 80)
        self.logger.info("BASELINE TRAINING (MATCHED TO LIGHTWEIGHT)")
        self.logger.info("=" * 80)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Architecture: ResNet34 + BERT + Attention Fusion")
        self.logger.info("=" * 80)
        
        # Create model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        print_model_summary(self.model)
        
        # Calculate class weights
        self.class_weights = self.calculate_class_weights(train_loader)
        self.class_weights = self.class_weights.to(self.device)
        
        # MATCHED optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,  # Now 0.007
            eps=1e-8
        )
        
        # MATCHED scheduler: ReduceLROnPlateau instead of CosineAnnealingWarmRestarts
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=4,
            min_lr=1e-7
        )
        
        # MATCHED loss function
        self.criterion = ImprovedFocalLoss(
            alpha=self.class_weights,
            gamma=self.focal_gamma  # Now 2.5
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"✅ Model initialized - {trainable_params:,} trainable parameters")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
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
                'architecture': 'baseline_attention_matched',
                'fusion_hidden_dim': self.fusion_hidden_dim,
                'num_attention_heads': self.num_attention_heads,
                'dropout': self.dropout,
                'focal_gamma': self.focal_gamma,
                'weight_decay': self.weight_decay,
            }
        }
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"🌟 Saved BEST model (Val Acc: {val_acc:.2f}%)")
        
        latest_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)
    
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
        plot_path = os.path.join(self.checkpoint_dir, 'training_curves_matched.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📊 Plots saved to {plot_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train the model."""
        start_time = datetime.now()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("MATCHED BASELINE TRAINING START")
        self.logger.info("=" * 80)
        
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"EPOCH {epoch}/{self.num_epochs}")
            self.logger.info(f"{'='*80}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            self.logger.info(f"✓ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.logger.info(f"✓ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Update learning rate
            self.scheduler.step(val_acc)  # CHANGED: Now based on val_acc
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
            else:
                self.save_checkpoint(epoch, val_loss, val_acc, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_acc):
                self.logger.info(f"\n⚠️ Early stopping at epoch {epoch}")
                break
        
        # Training completed
        duration = (datetime.now() - start_time).total_seconds()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("TRAINING COMPLETED!")
        self.logger.info("=" * 80)
        self.logger.info(f"Total time: {duration/60:.2f} minutes")
        self.logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        self.plot_training_history()


def main():
    """Main training with MATCHED parameters."""
    
    # MATCHED parameters to lightweight
    params = {
        'batch_size': 24,
        'learning_rate': 1e-4,
        'num_epochs': 30,  # CHANGED from 15
        'max_answer_vocab_size': 120,
        'fusion_hidden_dim': 384,  # CHANGED from 512
        'num_attention_heads': 6,  # CHANGED from 4
        'dropout': 0.35,  # CHANGED from 0.3
        'focal_gamma': 2.5,  # CHANGED from 2.0
        'weight_decay': 0.007,  # CHANGED from 0.01
        'early_stopping_patience': 10  # NEW
    }
    
    print("=" * 80)
    print("MATCHED BASELINE TRAINING CONFIGURATION")
    print("(Parameters matched to lightweight model for fair comparison)")
    print("=" * 80)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("=" * 80)
    
    # Load data
    print("\n📦 Loading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=params['batch_size'],
        max_samples=None,
        num_workers=0,
        max_answer_vocab_size=params['max_answer_vocab_size']
    )
    
    num_answers = len(answer_vocab)
    print(f"✓ Vocabulary: {num_answers} answers")
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = MatchedBaselineTrainer(
        num_answers=num_answers,
        **{k: v for k, v in params.items() if k not in ['max_answer_vocab_size']}
    )
    
    # Initialize model
    trainer.initialize_model(train_loader)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print("\nEvaluate with:")
    print("  python evaluate_with_roc.py --checkpoint checkpoints/baseline_matched/best_model.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
