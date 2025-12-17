#!/usr/bin/env python3
"""
Evaluation script for the lightweight attention model.
Mirrors baseline_attention evaluator outputs for consistency.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Resolve project root whether this file lives in repo root or evaluation/
PROJECT_ROOT = Path(__file__).resolve().parent
if not (PROJECT_ROOT / "checkpoints").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Redirect caches to a writable project-local directory (aligns with lightweight evaluator)
HF_CACHE_ROOT = PROJECT_ROOT / "hf_cache"
HF_DATASETS_CACHE = HF_CACHE_ROOT / "datasets"
HF_HUB_CACHE = HF_CACHE_ROOT / "hub"
MPL_CACHE = PROJECT_ROOT / "mpl_cache"

os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HUB_CACHE))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
MPL_CACHE.mkdir(parents=True, exist_ok=True)

from preprocessing.combined_preprocessing import create_combined_data_loaders
from models.lightweight_model import LightweightVQAModel


def _checkpoint_path() -> Path:
    """Return the default lightweight attention checkpoint path."""
    best_path = PROJECT_ROOT / "checkpoints"  / "recovery_training" / "best_model_recovered.pt"
    final_path = PROJECT_ROOT / "checkpoints" / "recovery_training" / "latest_model.pt"
    
    for path in (best_path, final_path):
        if path.exists():
            return path
    return best_path


class LightweightModelEvaluator:
    """Evaluator for lightweight attention VQA model - outputs match evaluate_saved_model."""
    
    def __init__(self,
                 checkpoint_path: str,
                 test_loader,
                 answer_vocab: dict,
                 device='cuda',
                 output_dir='evaluation_results/lightweight_improved_0812'):
        self.checkpoint_path = Path(checkpoint_path)
        self.test_loader = test_loader
        self.answer_vocab = answer_vocab
        self.idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
        self.device = device
        
        self.variant_name = self._extract_variant_name()
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Match evaluate_saved_model directory structure: evaluation_results/{architecture}/
        self.architecture = f'lightweight_{self.variant_name}'
        self.output_dir = Path(output_dir) / self.architecture
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store config and history from checkpoint
        self.config = {}
        self.history = {}
        
        self.model = self._load_model()
        
        print(f"✓ Loaded lightweight model: {self.variant_name}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def _extract_variant_name(self):
        """Extract variant name from checkpoint path."""
        parts = str(self.checkpoint_path).split('/')
        for part in parts:
            if part.startswith('lightweight_'):
                return part.replace('lightweight_', '')
        return 'attention'
    
    def _load_model(self):
        """Load lightweight model from checkpoint."""
        print(f"\nLoading checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Store config and history for later use
        self.config = checkpoint.get('config', {})
        self.history = checkpoint.get('history', {})
        
        model = LightweightVQAModel(
            num_classes=self.config.get('num_classes', self.config.get('num_answers', len(self.answer_vocab))),
            visual_feature_dim=self.config.get('visual_feature_dim', 576),
            text_feature_dim=self.config.get('text_feature_dim', 768),
            fusion_hidden_dim=self.config.get('fusion_hidden_dim', 256),
            num_attention_heads=self.config.get('num_attention_heads', 4),
            dropout=self.config.get('dropout', 0.3),
            freeze_vision_encoder=self.config.get('freeze_vision_encoder', False),
            freeze_text_encoder=self.config.get('freeze_text_encoder', False),
            use_spatial_features=self.config.get('use_spatial_features', False)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        params = model.count_parameters()
        print("  Architecture: MobileNetV3-Small + DistilBERT + attention fusion")
        print(f"  Total parameters: {params['total']:,}")
        print(f"  Model size: {model.get_model_size_mb():.2f} MB")
        
        return model
    
    @torch.no_grad()
    def _evaluate_split(self, data_loader, split_name: str):
        """Evaluate a specific split (validation/test) and save artifacts."""
        print("\n" + "=" * 80)
        print(f"EVALUATING ON {split_name.upper()} SET")
        print("=" * 80)
        
        all_predictions = []
        all_targets = []
        all_question_types = []
        all_question_texts = []
        all_answer_texts = []
        all_predicted_answers = []
        all_true_answers = []
        all_logits = []
        
        for batch in tqdm(data_loader, desc=f'Evaluating {split_name}'):
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            targets = batch['answer_idx'].to(self.device)
            
            logits = self.model(images, input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_question_types.extend(batch['question_type'])
            all_question_texts.extend(batch['question']['text'])
            all_answer_texts.extend(batch['answer']['text'])
            
            for pred_idx, target_idx in zip(predictions.cpu().numpy(), targets.cpu().numpy()):
                all_predicted_answers.append(self.idx_to_answer.get(pred_idx, '<unk>'))
                all_true_answers.append(self.idx_to_answer.get(target_idx, '<unk>'))
        
        results = self._calculate_metrics(
            all_predictions, all_targets, all_question_types,
            all_logits, all_predicted_answers, all_true_answers
        )
        
        # Save artifacts matching evaluate_saved_model naming conventions
        self._save_detailed_results(
            all_predictions, all_targets, all_question_types,
            all_question_texts, all_answer_texts,
            all_predicted_answers, all_true_answers,
            split_name=split_name
        )
        
        self._plot_confusion_matrix(
            all_predictions, all_targets,
            split_name=split_name
        )
        
        self._plot_per_type_accuracy(
            all_predictions, all_targets, all_question_types,
            split_name=split_name
        )
        
        self._print_summary(results)
        
        return results
    
    def _calculate_metrics(self, predictions, targets, question_types, logits, pred_answers, true_answers):
        """Calculate comprehensive metrics matching evaluate_saved_model format."""
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(targets, predictions)
        results['total_samples'] = len(targets)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        results['macro_precision'] = precision
        results['macro_recall'] = recall
        results['macro_f1'] = f1
        
        # Binary metrics (yes/no questions)
        binary_answers = {'yes', 'no'}
        binary_mask = [ans in binary_answers for ans in pred_answers]
        if sum(binary_mask) > 0:
            binary_preds = [p for p, m in zip(predictions, binary_mask) if m]
            binary_targets = [t for t, m in zip(targets, binary_mask) if m]
            
            results['binary_accuracy'] = accuracy_score(binary_targets, binary_preds)
            results['binary_samples'] = sum(binary_mask)
            
            if len(set(binary_targets)) > 1:
                b_precision, b_recall, b_f1, _ = precision_recall_fscore_support(
                    binary_targets, binary_preds, average='macro', zero_division=0, pos_label=1
                )
                results['binary_precision'] = b_precision
                results['binary_recall'] = b_recall
                results['binary_f1'] = b_f1
        else:
            # Initialize binary metrics to 0 if no binary questions
            results['binary_accuracy'] = 0
            results['binary_samples'] = 0
            results['binary_precision'] = 0
            results['binary_recall'] = 0
            results['binary_f1'] = 0
        
        # Per question type
        results['by_question_type'] = {}
        for qtype in set(question_types):
            mask = [qt == qtype for qt in question_types]
            if sum(mask) > 0:
                type_preds = [p for p, m in zip(predictions, mask) if m]
                type_targets = [t for t, m in zip(targets, mask) if m]
                
                results['by_question_type'][qtype] = {
                    'accuracy': accuracy_score(type_targets, type_preds),
                    'samples': sum(mask)
                }
        
        # Top-K accuracy
        logits_array = np.array(logits)
        targets_array = np.array(targets)
        
        for k in [3, 5, 10]:
            top_k_preds = np.argsort(logits_array, axis=1)[:, -k:]
            top_k_correct = [t in top_k_preds[i] for i, t in enumerate(targets_array)]
            results[f'top_{k}_accuracy'] = np.mean(top_k_correct)
        
        return results
    
    def _save_detailed_results(self, predictions, targets, question_types,
                               question_texts, answer_texts, pred_answers, true_answers,
                               split_name: str = "test"):
        """Save detailed results to CSV - matching evaluate_saved_model format."""
        import pandas as pd
        
        df = pd.DataFrame({
            'question': question_texts,
            'true_answer': answer_texts,
            'predicted_answer': pred_answers,
            'correct': [p == t for p, t in zip(predictions, targets)],
            'question_type': question_types
        })
        
        # Primary file: {split}_detailed_{timestamp}.csv (matches evaluate_saved_model)
        detailed_name = f'{split_name}_detailed_{self.timestamp}.csv'
        csv_path = self.output_dir / detailed_name
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Detailed results saved to: {csv_path}")
    
    def _plot_confusion_matrix(self, predictions, targets, split_name: str = "test"):
        """Generate confusion matrix plot - matching evaluate_saved_model naming."""
        from collections import Counter
        
        target_counts = Counter(targets)
        top_classes = [cls for cls, _ in target_counts.most_common(20)]
        
        mask = [t in top_classes for t in targets]
        if sum(mask) > 10:
            filtered_preds = [p for p, m in zip(predictions, mask) if m]
            filtered_targets = [t for t, m in zip(targets, mask) if m]
            
            cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            
            class_labels = [self.idx_to_answer.get(cls, f'class_{cls}') for cls in top_classes]
            ax.set_xticklabels(class_labels, rotation=45, ha='right')
            ax.set_yticklabels(class_labels, rotation=0)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - {split_name.capitalize()} Set')
            
            plt.tight_layout()
            # Match evaluate_saved_model naming: {split}_confusion_matrix_{timestamp}.png
            cm_path = self.output_dir / f'{split_name}_confusion_matrix_{self.timestamp}.png'
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Confusion matrix saved to: {cm_path}")
    
    def _plot_per_type_accuracy(self, predictions, targets, question_types, split_name: str = "test"):
        """Generate per-type accuracy plot - matching evaluate_saved_model naming."""
        type_accuracies = {}
        type_counts = {}
        
        for qtype in set(question_types):
            mask = [qt == qtype for qt in question_types]
            if sum(mask) > 0:
                type_preds = [p for p, m in zip(predictions, mask) if m]
                type_targets = [t for t, m in zip(targets, mask) if m]
                type_accuracies[qtype] = accuracy_score(type_targets, type_preds)
                type_counts[qtype] = sum(mask)
        
        if type_accuracies:
            fig, ax = plt.subplots(figsize=(10, 6))
            types = list(type_accuracies.keys())
            accs = [type_accuracies[t] for t in types]
            counts = [type_counts[t] for t in types]
            
            bars = ax.bar(types, accs, color='steelblue', alpha=0.7)
            
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'n={count}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Question Type')
            ax.set_title(f'Accuracy by Question Type - {split_name.capitalize()} Set')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            # Match evaluate_saved_model naming: {split}_per_type_{timestamp}.png
            per_type_path = self.output_dir / f'{split_name}_per_type_{self.timestamp}.png'
            plt.savefig(per_type_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Per-type accuracy plot saved to: {per_type_path}")
    
    def _print_summary(self, results):
        """Print summary matching evaluate_saved_model format."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Total samples: {results['total_samples']}")
        
        print(f"\nMulti-class Metrics:")
        print(f"  Macro Precision: {results['macro_precision']:.4f}")
        print(f"  Macro Recall:    {results['macro_recall']:.4f}")
        print(f"  Macro F1-Score:  {results['macro_f1']:.4f}")
        
        print(f"\nTop-K Accuracy:")
        print(f"  Top-3:  {results['top_3_accuracy']:.4f}")
        print(f"  Top-5:  {results['top_5_accuracy']:.4f}")
        print(f"  Top-10: {results['top_10_accuracy']:.4f}")
        
        if results.get('binary_samples', 0) > 0:
            print(f"\nBinary Questions (Yes/No):")
            print(f"  Accuracy:  {results['binary_accuracy']:.4f}")
            print(f"  Precision: {results['binary_precision']:.4f}")
            print(f"  Recall:    {results['binary_recall']:.4f}")
            print(f"  F1-Score:  {results['binary_f1']:.4f}")
            print(f"  Samples:   {results['binary_samples']}")
        
        if results.get('by_question_type'):
            print(f"\nBy Question Type:")
            for qtype, metrics in results['by_question_type'].items():
                print(f"  {qtype}:")
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    Samples:  {metrics['samples']}")
        
        print("=" * 80)
    
    def full_evaluation(self, val_loader=None, test_loader=None):
        """
        Perform full evaluation matching evaluate_saved_model structure.
        Generates: metrics_summary_{timestamp}.json and evaluation_report_{timestamp}.txt
        """
        metrics_summary = {
            'checkpoint_path': str(self.checkpoint_path),
            'architecture': self.architecture,
            'device': str(self.device),
            'timestamp': self.timestamp
        }
        
        # Evaluate validation set
        if val_loader is not None:
            print("\n" + "="*80)
            print("VALIDATION SET EVALUATION")
            print("="*80)
            val_results = self._evaluate_split(val_loader, 'validation')
            metrics_summary['validation'] = val_results
        
        # Evaluate test set
        if test_loader is not None:
            print("\n" + "="*80)
            print("TEST SET EVALUATION")
            print("="*80)
            test_results = self._evaluate_split(test_loader, 'test')
            metrics_summary['test'] = test_results
        
        # Save metrics_summary_{timestamp}.json (matches evaluate_saved_model)
        metrics_path = self.output_dir / f'metrics_summary_{self.timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\n✓ Metrics summary saved to: {metrics_path}")
        
        # Generate evaluation_report_{timestamp}.txt (matches evaluate_saved_model)
        self._generate_evaluation_report(metrics_summary)
        
        return metrics_summary
    
    def _generate_evaluation_report(self, metrics_summary: dict):
        """Generate comprehensive text report matching evaluate_saved_model format."""
        report_path = self.output_dir / f'evaluation_report_{self.timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VQA MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {metrics_summary['checkpoint_path']}\n")
            f.write(f"Architecture: {metrics_summary['architecture']}\n")
            f.write(f"Device: {metrics_summary['device']}\n\n")
            
            # Add model configuration section
            f.write("=" * 80 + "\n")
            f.write("MODEL CONFIGURATION\n")
            f.write("=" * 80 + "\n")
            if self.config:
                for key, value in sorted(self.config.items()):
                    f.write(f"{key}: {value}\n")
            else:
                f.write("(No configuration available)\n")
            f.write("\n")
            
            # Validation results
            if 'validation' in metrics_summary:
                f.write("=" * 80 + "\n")
                f.write("VALIDATION SET RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                val = metrics_summary['validation']
                f.write(f"Overall Accuracy: {val['accuracy']:.4f} ({val['accuracy']*100:.2f}%)\n\n")
                
                if val.get('binary_samples', 0) > 0:
                    f.write("Binary Questions (Yes/No):\n")
                    f.write(f"  Accuracy:  {val['binary_accuracy']:.4f}\n")
                    f.write(f"  Precision: {val['binary_precision']:.4f}\n")
                    f.write(f"  Recall:    {val['binary_recall']:.4f}\n")
                    f.write(f"  F1-Score:  {val['binary_f1']:.4f}\n\n")
                
                f.write("Multi-class Metrics:\n")
                f.write(f"  Macro Precision: {val['macro_precision']:.4f}\n")
                f.write(f"  Macro Recall:    {val['macro_recall']:.4f}\n")
                f.write(f"  Macro F1-Score:  {val['macro_f1']:.4f}\n\n")
            
            # Test results
            if 'test' in metrics_summary:
                f.write("=" * 80 + "\n")
                f.write("TEST SET RESULTS\n")
                f.write("=" * 80 + "\n\n")
                
                test = metrics_summary['test']
                f.write(f"Overall Accuracy: {test['accuracy']:.4f} ({test['accuracy']*100:.2f}%)\n\n")
                
                if test.get('binary_samples', 0) > 0:
                    f.write("Binary Questions (Yes/No):\n")
                    f.write(f"  Accuracy:  {test['binary_accuracy']:.4f}\n")
                    f.write(f"  Precision: {test['binary_precision']:.4f}\n")
                    f.write(f"  Recall:    {test['binary_recall']:.4f}\n")
                    f.write(f"  F1-Score:  {test['binary_f1']:.4f}\n\n")
                
                f.write("Multi-class Metrics:\n")
                f.write(f"  Macro Precision: {test['macro_precision']:.4f}\n")
                f.write(f"  Macro Recall:    {test['macro_recall']:.4f}\n")
                f.write(f"  Macro F1-Score:  {test['macro_f1']:.4f}\n\n")
            
            # Add training history section
            f.write("=" * 80 + "\n")
            f.write("TRAINING HISTORY (from checkpoint)\n")
            f.write("=" * 80 + "\n\n")
            
            if self.history:
                f.write(f"Training Epochs: {len(self.history.get('train_loss', []))}\n")
                if self.history.get('train_loss'):
                    f.write(f"Final Training Loss: {self.history['train_loss'][-1]:.4f}\n")
                    f.write(f"Best Training Loss:  {min(self.history['train_loss']):.4f}\n")
                if self.history.get('train_acc'):
                    f.write(f"Final Training Acc:  {self.history['train_acc'][-1]:.2f}%\n")
                    f.write(f"Best Training Acc:   {max(self.history['train_acc']):.2f}%\n")
                if self.history.get('val_loss'):
                    f.write(f"Final Val Loss:      {self.history['val_loss'][-1]:.4f}\n")
                    f.write(f"Best Val Loss:       {min(self.history['val_loss']):.4f}\n")
                if self.history.get('val_acc'):
                    f.write(f"Final Val Acc:       {self.history['val_acc'][-1]:.2f}%\n")
                    f.write(f"Best Val Acc:        {max(self.history['val_acc']):.2f}%\n")
            else:
                f.write("(No training history available)\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✓ Evaluation report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Lightweight Attention VQA Model')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to a checkpoint (defaults to lightweight_attention/checkpoint_best.pth)')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--max_answer_vocab_size', type=int, default=120)
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    print("=" * 80)
    print("VQA BASELINE MODEL EVALUATION")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        max_answer_vocab_size=args.max_answer_vocab_size,
        encode_answers=False
    )
    
    # Determine checkpoint path
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else _checkpoint_path()
    
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return
    
    # Create evaluator and run full evaluation
    evaluator = LightweightModelEvaluator(
        checkpoint_path, 
        test_loader, 
        answer_vocab, 
        args.device, 
        args.output_dir
    )
    
    # Run full evaluation (generates all output files)
    metrics_summary = evaluator.full_evaluation(
        val_loader=val_loader,
        test_loader=test_loader
    )
    
    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETED!")
    print("=" * 80)
    print(f"\nGenerated files in {evaluator.output_dir}/:")
    print("  📊 Detailed results (CSV)")
    print("  📈 Confusion matrices (PNG)")
    print("  📉 Per-type accuracy plots (PNG)")
    print("  📄 Evaluation report (TXT)")
    print("  📋 Metrics summary (JSON)")
    print("=" * 80)


if __name__ == "__main__":
    main()
