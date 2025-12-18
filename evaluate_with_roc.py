#!/usr/bin/env python3
"""
Enhanced evaluation script with AUC-ROC metrics and ROC curves.
Can be used for any VQA model checkpoint.
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
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from combined_preprocessing import create_combined_data_loaders
from lightweight_model import LightweightVQAModel
from baseline_model_attention import BaselineVQAModel


class EnhancedModelEvaluator:
    """Evaluator with AUC-ROC metrics and ROC curves."""
    
    def __init__(self,
                 checkpoint_path: str,
                 data_loader,
                 answer_vocab: dict,
                 device='cuda',
                 output_dir=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.data_loader = data_loader
        self.answer_vocab = answer_vocab
        self.idx_to_answer = {idx: ans for ans, idx in answer_vocab.items()}
        self.num_classes = len(answer_vocab)
        self.device = device
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_dir is None:
            output_dir = f'evaluation_results_roc/{self.timestamp}'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = {}
        self.history = {}
        
        self.model = self._load_model()
        
        print(f"✓ Loaded model from: {self.checkpoint_path}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if 'config' in checkpoint:
            self.config = checkpoint['config']
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        # Detect model type
        architecture = self.config.get('architecture', 'unknown')
        
        if 'lightweight' in architecture or 'mobile' in architecture.lower():
            model = LightweightVQAModel(
                num_classes=self.num_classes,
                fusion_hidden_dim=self.config.get('fusion_hidden_dim', 256),
                num_attention_heads=self.config.get('num_attention_heads', 4),
                dropout=self.config.get('dropout', 0.3),
                freeze_vision_encoder=False,
                freeze_text_encoder=False
            )
        else:
            # Baseline model
            model = BaselineVQAModel(
                num_classes=self.num_classes,
                dropout=self.config.get('dropout', 0.3)
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    @torch.no_grad()
    def evaluate(self):
        """Run evaluation with AUC-ROC metrics."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []  # For AUC-ROC
        all_logits = []
        all_question_types = []
        all_question_texts = []
        all_answer_texts = []
        
        print("\n" + "="*80)
        print("RUNNING EVALUATION WITH AUC-ROC METRICS")
        print("="*80)
        
        for batch in tqdm(self.data_loader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            input_ids = batch['question']['input_ids'].to(self.device)
            attention_mask = batch['question']['attention_mask'].to(self.device)
            answers = batch['answer_idx'].to(self.device)
            
            logits = self.model(images, input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(answers.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_question_types.extend(batch.get('question_type', ['unknown'] * len(answers)))
            all_question_texts.extend(batch.get('question_text', [''] * len(answers)))
            all_answer_texts.extend([self.idx_to_answer.get(a.item(), 'unknown') 
                                    for a in answers])
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_logits = np.array(all_logits)
        
        pred_answers = [self.idx_to_answer.get(p, 'unknown') for p in all_predictions]
        
        # Calculate metrics
        results = self._calculate_metrics_with_roc(
            all_predictions, all_targets, all_probabilities,
            all_question_types, pred_answers, all_answer_texts
        )
        
        # Save detailed results
        self._save_detailed_results(
            all_predictions, all_targets, all_probabilities,
            all_question_types, all_question_texts, 
            all_answer_texts, pred_answers
        )
        
        # Generate visualizations
        self._plot_confusion_matrix(all_predictions, all_targets)
        self._plot_roc_curves(all_targets, all_probabilities)
        self._plot_per_class_auc(all_targets, all_probabilities)
        
        # Print and save summary
        self._print_summary(results)
        self._save_summary(results)
        
        return results
    
    def _calculate_metrics_with_roc(self, predictions, targets, probabilities, 
                                     question_types, pred_answers, true_answers):
        """Calculate comprehensive metrics including AUC-ROC."""
        results = {}
        
        # Overall metrics
        results['accuracy'] = accuracy_score(targets, predictions)
        results['total_samples'] = len(targets)
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        results['macro_precision'] = precision
        results['macro_recall'] = recall
        results['macro_f1'] = f1
        
        # AUC-ROC Metrics
        try:
            # Multi-class AUC (One-vs-Rest)
            # Only calculate for classes present in test set
            unique_classes = np.unique(targets)
            
            if len(unique_classes) > 1:
                # Select only probabilities for classes present in test set
                present_probs = probabilities[:, unique_classes]
                # Renormalize so probabilities sum to 1.0
                present_probs = present_probs / present_probs.sum(axis=1, keepdims=True)
                
                results['auc_ovr_macro'] = roc_auc_score(
                    targets, present_probs, 
                    multi_class='ovr', 
                    average='macro',
                    labels=unique_classes
                )
                results['auc_ovr_weighted'] = roc_auc_score(
                    targets, present_probs,
                    multi_class='ovr',
                    average='weighted',
                    labels=unique_classes
                )
            else:
                print(f"⚠️  Only one class in test set, cannot calculate multi-class AUC")
                results['auc_ovr_macro'] = 0.0
                results['auc_ovr_weighted'] = 0.0
        except Exception as e:
            print(f"⚠️  Could not calculate multi-class AUC: {e}")
            results['auc_ovr_macro'] = 0.0
            results['auc_ovr_weighted'] = 0.0
        
        # Binary metrics (yes/no questions)
        binary_answers = {'yes', 'no'}
        binary_mask = [ans in binary_answers for ans in pred_answers]
        
        if sum(binary_mask) > 0:
            binary_preds = predictions[binary_mask]
            binary_targets = targets[binary_mask]
            binary_probs = probabilities[binary_mask]
            
            results['binary_accuracy'] = accuracy_score(binary_targets, binary_preds)
            results['binary_samples'] = sum(binary_mask)
            
            if len(set(binary_targets)) > 1:
                b_precision, b_recall, b_f1, _ = precision_recall_fscore_support(
                    binary_targets, binary_preds, average='macro', zero_division=0
                )
                results['binary_precision'] = b_precision
                results['binary_recall'] = b_recall
                results['binary_f1'] = b_f1
                
                # Binary AUC-ROC
                try:
                    # Get yes/no class indices
                    yes_idx = self.answer_vocab.get('yes', -1)
                    no_idx = self.answer_vocab.get('no', -1)
                    
                    if yes_idx != -1 and no_idx != -1:
                        # Binary classification: use probability of positive class
                        binary_labels = (binary_targets == yes_idx).astype(int)
                        binary_prob_positive = binary_probs[:, yes_idx]
                        
                        results['binary_auc'] = roc_auc_score(
                            binary_labels, binary_prob_positive
                        )
                except Exception as e:
                    print(f"⚠️  Could not calculate binary AUC: {e}")
                    results['binary_auc'] = 0.0
        else:
            results['binary_accuracy'] = 0
            results['binary_samples'] = 0
            results['binary_precision'] = 0
            results['binary_recall'] = 0
            results['binary_f1'] = 0
            results['binary_auc'] = 0
        
        # Per-class AUC
        results['per_class_auc'] = {}
        target_binarized = label_binarize(targets, classes=np.arange(self.num_classes))
        
        for class_idx in range(self.num_classes):
            class_name = self.idx_to_answer.get(class_idx, f'class_{class_idx}')
            
            # Only calculate if this class appears in targets
            if target_binarized[:, class_idx].sum() > 0:
                try:
                    class_auc = roc_auc_score(
                        target_binarized[:, class_idx],
                        probabilities[:, class_idx]
                    )
                    results['per_class_auc'][class_name] = class_auc
                except:
                    results['per_class_auc'][class_name] = 0.0
        
        # Per question type
        results['by_question_type'] = {}
        for qtype in set(question_types):
            mask = np.array([qt == qtype for qt in question_types])
            if sum(mask) > 0:
                type_preds = predictions[mask]
                type_targets = targets[mask]
                
                results['by_question_type'][qtype] = {
                    'accuracy': accuracy_score(type_targets, type_preds),
                    'samples': sum(mask)
                }
        
        # Top-K accuracy
        for k in [3, 5, 10]:
            top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
            top_k_correct = [t in top_k_preds[i] for i, t in enumerate(targets)]
            results[f'top_{k}_accuracy'] = np.mean(top_k_correct)
        
        return results
    
    def _plot_roc_curves(self, targets, probabilities):
        """Plot ROC curves for top classes and macro-average."""
        from collections import Counter
        
        # Get top 10 most frequent classes
        target_counts = Counter(targets)
        top_classes = [cls for cls, _ in target_counts.most_common(10)]
        
        # Binarize targets
        target_binarized = label_binarize(targets, classes=np.arange(self.num_classes))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top 5 classes
        ax1 = axes[0, 0]
        for class_idx in top_classes[:5]:
            if target_binarized[:, class_idx].sum() > 0:
                fpr, tpr, _ = roc_curve(
                    target_binarized[:, class_idx],
                    probabilities[:, class_idx]
                )
                roc_auc = auc(fpr, tpr)
                class_name = self.idx_to_answer.get(class_idx, f'Class {class_idx}')
                ax1.plot(fpr, tpr, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curves - Top 5 Classes', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Next 5 classes
        ax2 = axes[0, 1]
        for class_idx in top_classes[5:10]:
            if target_binarized[:, class_idx].sum() > 0:
                fpr, tpr, _ = roc_curve(
                    target_binarized[:, class_idx],
                    probabilities[:, class_idx]
                )
                roc_auc = auc(fpr, tpr)
                class_name = self.idx_to_answer.get(class_idx, f'Class {class_idx}')
                ax2.plot(fpr, tpr, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax2.set_xlabel('False Positive Rate', fontsize=11)
        ax2.set_ylabel('True Positive Rate', fontsize=11)
        ax2.set_title('ROC Curves - Classes 6-10', fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Binary (Yes/No) if available
        ax3 = axes[1, 0]
        yes_idx = self.answer_vocab.get('yes', -1)
        no_idx = self.answer_vocab.get('no', -1)
        
        if yes_idx != -1 and target_binarized[:, yes_idx].sum() > 0:
            fpr, tpr, _ = roc_curve(
                target_binarized[:, yes_idx],
                probabilities[:, yes_idx]
            )
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, lw=3, label=f'Yes (AUC = {roc_auc:.3f})', color='green')
        
        if no_idx != -1 and target_binarized[:, no_idx].sum() > 0:
            fpr, tpr, _ = roc_curve(
                target_binarized[:, no_idx],
                probabilities[:, no_idx]
            )
            roc_auc = auc(fpr, tpr)
            ax3.plot(fpr, tpr, lw=3, label=f'No (AUC = {roc_auc:.3f})', color='red')
        
        ax3.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax3.set_xlabel('False Positive Rate', fontsize=11)
        ax3.set_ylabel('True Positive Rate', fontsize=11)
        ax3.set_title('ROC Curves - Binary Questions (Yes/No)', fontsize=13, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Macro-average ROC
        ax4 = axes[1, 1]
        
        # Compute macro-average ROC
        all_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(all_fpr)
        
        for class_idx in range(self.num_classes):
            if target_binarized[:, class_idx].sum() > 0:
                fpr, tpr, _ = roc_curve(
                    target_binarized[:, class_idx],
                    probabilities[:, class_idx]
                )
                mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        mean_tpr /= self.num_classes
        macro_auc = auc(all_fpr, mean_tpr)
        
        ax4.plot(all_fpr, mean_tpr, lw=3, 
                label=f'Macro-average (AUC = {macro_auc:.3f})',
                color='navy')
        ax4.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax4.set_xlabel('False Positive Rate', fontsize=11)
        ax4.set_ylabel('True Positive Rate', fontsize=11)
        ax4.set_title('Macro-Average ROC Curve', fontsize=13, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'roc_curves_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to: {plot_path}")
    
    def _plot_per_class_auc(self, targets, probabilities):
        """Plot per-class AUC scores."""
        from collections import Counter
        
        target_counts = Counter(targets)
        top_classes = [cls for cls, _ in target_counts.most_common(20)]
        
        target_binarized = label_binarize(targets, classes=np.arange(self.num_classes))
        
        class_aucs = []
        class_names = []
        
        for class_idx in top_classes:
            if target_binarized[:, class_idx].sum() > 0:
                try:
                    class_auc = roc_auc_score(
                        target_binarized[:, class_idx],
                        probabilities[:, class_idx]
                    )
                    class_aucs.append(class_auc)
                    class_names.append(self.idx_to_answer.get(class_idx, f'Class {class_idx}'))
                except:
                    pass
        
        # Sort by AUC
        sorted_pairs = sorted(zip(class_names, class_aucs), key=lambda x: x[1], reverse=True)
        class_names, class_aucs = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.RdYlGn(np.array(class_aucs))
        bars = ax.barh(range(len(class_names)), class_aucs, color=colors)
        
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names)
        ax.set_xlabel('AUC-ROC Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class AUC-ROC Scores (Top 20)', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, class_aucs)):
            ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.output_dir / f'per_class_auc_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class AUC plot saved to: {plot_path}")
    
    def _plot_confusion_matrix(self, predictions, targets):
        """Plot confusion matrix for top classes."""
        from collections import Counter
        
        target_counts = Counter(targets)
        top_classes = [cls for cls, _ in target_counts.most_common(15)]
        
        mask = np.isin(targets, top_classes)
        if sum(mask) > 10:
            filtered_preds = predictions[mask]
            filtered_targets = targets[mask]
            
            cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
            
            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=[self.idx_to_answer.get(c, f'{c}') for c in top_classes],
                       yticklabels=[self.idx_to_answer.get(c, f'{c}') for c in top_classes])
            
            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
            ax.set_ylabel('True', fontsize=12, fontweight='bold')
            ax.set_title('Confusion Matrix (Top 15 Classes)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plot_path = self.output_dir / f'confusion_matrix_{self.timestamp}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Confusion matrix saved to: {plot_path}")
    
    def _save_detailed_results(self, predictions, targets, probabilities,
                               question_types, question_texts, 
                               true_answers, pred_answers):
        """Save detailed results to CSV."""
        import pandas as pd
        
        # Get confidence scores (max probability)
        confidences = np.max(probabilities, axis=1)
        
        df = pd.DataFrame({
            'question': question_texts,
            'true_answer': true_answers,
            'predicted_answer': pred_answers,
            'confidence': confidences,
            'correct': predictions == targets,
            'question_type': question_types,
            'true_label': targets,
            'pred_label': predictions
        })
        
        csv_path = self.output_dir / f'detailed_results_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to: {csv_path}")
    
    def _print_summary(self, results):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS WITH AUC-ROC")
        print("="*80)
        
        print(f"\nOverall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Total Samples: {results['total_samples']}")
        
        print(f"\nMulti-class Metrics:")
        print(f"  Macro Precision: {results['macro_precision']:.4f}")
        print(f"  Macro Recall:    {results['macro_recall']:.4f}")
        print(f"  Macro F1-Score:  {results['macro_f1']:.4f}")
        
        print(f"\nAUC-ROC Scores:")
        print(f"  Macro-average:    {results['auc_ovr_macro']:.4f}")
        print(f"  Weighted-average: {results['auc_ovr_weighted']:.4f}")
        
        if results['binary_samples'] > 0:
            print(f"\nBinary Questions (Yes/No):")
            print(f"  Accuracy:  {results['binary_accuracy']:.4f}")
            print(f"  Precision: {results['binary_precision']:.4f}")
            print(f"  Recall:    {results['binary_recall']:.4f}")
            print(f"  F1-Score:  {results['binary_f1']:.4f}")
            print(f"  AUC-ROC:   {results.get('binary_auc', 0):.4f}")
        
        print(f"\nTop-K Accuracy:")
        print(f"  Top-3:  {results['top_3_accuracy']:.4f}")
        print(f"  Top-5:  {results['top_5_accuracy']:.4f}")
        print(f"  Top-10: {results['top_10_accuracy']:.4f}")
        
        print("\n" + "="*80)
    
    def _save_summary(self, results):
        """Save evaluation summary to file."""
        summary_path = self.output_dir / f'evaluation_summary_{self.timestamp}.json'
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✓ Summary saved to: {summary_path}")
        
        # Also save as text
        text_path = self.output_dir / f'evaluation_report_{self.timestamp}.txt'
        with open(text_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("VQA MODEL EVALUATION REPORT (WITH AUC-ROC)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Checkpoint: {self.checkpoint_path}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n\n")
            
            f.write("Multi-class Metrics:\n")
            f.write(f"  Macro Precision: {results['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall:    {results['macro_recall']:.4f}\n")
            f.write(f"  Macro F1-Score:  {results['macro_f1']:.4f}\n\n")
            
            f.write("AUC-ROC Scores:\n")
            f.write(f"  Macro-average:    {results['auc_ovr_macro']:.4f}\n")
            f.write(f"  Weighted-average: {results['auc_ovr_weighted']:.4f}\n\n")
            
            if results['binary_samples'] > 0:
                f.write("Binary Questions (Yes/No):\n")
                f.write(f"  Accuracy:  {results['binary_accuracy']:.4f}\n")
                f.write(f"  Precision: {results['binary_precision']:.4f}\n")
                f.write(f"  Recall:    {results['binary_recall']:.4f}\n")
                f.write(f"  F1-Score:  {results['binary_f1']:.4f}\n")
                f.write(f"  AUC-ROC:   {results.get('binary_auc', 0):.4f}\n\n")
            
            f.write("="*80 + "\n")
        
        print(f"✓ Text report saved to: {text_path}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced VQA model evaluation with AUC-ROC')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=24,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/mps/cpu)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, answer_vocab, _ = create_combined_data_loaders(
        batch_size=args.batch_size,
        num_workers=0,
        max_answer_vocab_size=120
    )
    
    # Select split
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    print(f"✓ Loaded {args.split} split with {len(data_loader)} batches")
    print(f"✓ Answer vocabulary size: {len(answer_vocab)}")
    
    # Run evaluation
    evaluator = EnhancedModelEvaluator(
        checkpoint_path=args.checkpoint,
        data_loader=data_loader,
        answer_vocab=answer_vocab,
        device=device,
        output_dir=args.output_dir
    )
    
    results = evaluator.evaluate()
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {evaluator.output_dir}")


if __name__ == '__main__':
    main()

