#!/usr/bin/env python3
"""
Evaluation Metrics for VQA-RAD Dataset
Comprehensive evaluation including accuracy, precision, recall, F1, and AUC-ROC
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class VQAEvaluator:
    """Comprehensive evaluator for VQA models."""
    
    def __init__(self, answer_vocab: Dict[str, int], question_types: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            answer_vocab: Dictionary mapping answer strings to indices
            question_types: List of question types in the dataset
        """
        self.answer_vocab = answer_vocab
        self.idx_to_answer = {v: k for k, v in answer_vocab.items()}
        self.question_types = question_types or ['binary', 'open-ended']
        
        # Storage for predictions and ground truth
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and ground truth."""
        self.all_predictions = []
        self.all_ground_truth = []
        self.all_probabilities = []
        self.all_question_types = []
        self.all_questions = []
        self.all_answers_text = []
    
    def add_batch(self, 
                  predictions: torch.Tensor,
                  ground_truth: torch.Tensor,
                  probabilities: Optional[torch.Tensor] = None,
                  question_types: Optional[List[str]] = None,
                  questions: Optional[List[str]] = None,
                  answers: Optional[List[str]] = None):
        """
        Add a batch of predictions for evaluation.
        
        Args:
            predictions: Predicted answer indices [batch_size]
            ground_truth: Ground truth answer indices [batch_size]
            probabilities: Prediction probabilities [batch_size, num_classes]
            question_types: List of question types for this batch
            questions: List of question texts
            answers: List of answer texts
        """
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_ground_truth.extend(ground_truth.cpu().numpy())
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())
        
        if question_types is not None:
            self.all_question_types.extend(question_types)
        
        if questions is not None:
            self.all_questions.extend(questions)
        
        if answers is not None:
            self.all_answers_text.extend(answers)
    
    def compute_accuracy(self) -> float:
        """Compute overall accuracy."""
        predictions = np.array(self.all_predictions)
        ground_truth = np.array(self.all_ground_truth)
        return accuracy_score(ground_truth, predictions)
    
    def compute_binary_metrics(self) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for yes/no questions if present,
        otherwise returns zeros.
        """
        # Filter for binary questions (yes/no)
        binary_mask = []
        binary_answers = set()
        for i, ans_idx in enumerate(self.all_ground_truth):
            answer = self.idx_to_answer.get(ans_idx, '').lower()
            if answer in ['yes', 'no']:
                binary_mask.append(i)
                binary_answers.add(answer)
        
        # If we don't have both yes and no answers, or no binary questions
        if len(binary_mask) == 0 or len(binary_answers) < 2:
            return {
                'binary_accuracy': 0.0,
                'binary_precision': 0.0,
                'binary_recall': 0.0,
                'binary_f1': 0.0
            }
        
        # Extract binary predictions
        binary_preds = [self.all_predictions[i] for i in binary_mask]
        binary_gt = [self.all_ground_truth[i] for i in binary_mask]
        
        return {
            'binary_accuracy': accuracy_score(binary_gt, binary_preds),
            'binary_precision': precision_score(binary_gt, binary_preds, average='macro', zero_division=0),
            'binary_recall': recall_score(binary_gt, binary_preds, average='macro', zero_division=0),
            'binary_f1': f1_score(binary_gt, binary_preds, average='macro', zero_division=0)
        }
    
    def compute_multiclass_metrics(self) -> Dict[str, float]:
        """Compute precision, recall, F1 for multi-class classification."""
        predictions = np.array(self.all_predictions)
        ground_truth = np.array(self.all_ground_truth)
        
        return {
            'macro_precision': precision_score(ground_truth, predictions, average='macro', zero_division=0),
            'macro_recall': recall_score(ground_truth, predictions, average='macro', zero_division=0),
            'macro_f1': f1_score(ground_truth, predictions, average='macro', zero_division=0),
            'weighted_precision': precision_score(ground_truth, predictions, average='weighted', zero_division=0),
            'weighted_recall': recall_score(ground_truth, predictions, average='weighted', zero_division=0),
            'weighted_f1': f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        }
    
    def compute_auc_roc(self) -> Dict[str, float]:
        """
        Compute AUC-ROC scores.
        For binary: single AUC-ROC
        For multi-class: one-vs-rest AUC-ROC
        """
        if len(self.all_probabilities) == 0:
            return {'auc_roc': 0.0}
        
        probabilities = np.array(self.all_probabilities)
        ground_truth = np.array(self.all_ground_truth)
        
        results = {}
        
        # Binary AUC-ROC
        binary_mask = []
        for i, ans_idx in enumerate(ground_truth):
            answer = self.idx_to_answer.get(ans_idx, '').lower()
            if answer in ['yes', 'no']:
                binary_mask.append(i)
        
        if len(binary_mask) > 0 and len(set([ground_truth[i] for i in binary_mask])) > 1:
            binary_gt = [ground_truth[i] for i in binary_mask]
            binary_probs = probabilities[binary_mask]
            
            # Get probability of positive class
            yes_idx = [idx for ans, idx in self.answer_vocab.items() if ans.lower() == 'yes']
            if len(yes_idx) > 0:
                binary_probs_pos = binary_probs[:, yes_idx[0]]
                results['binary_auc_roc'] = roc_auc_score(binary_gt, binary_probs_pos)
        
        # Multi-class AUC-ROC (one-vs-rest)
        try:
            # Only compute if we have enough classes
            unique_classes = np.unique(ground_truth)
            if len(unique_classes) > 1:
                results['macro_auc_roc'] = roc_auc_score(
                    ground_truth, 
                    probabilities, 
                    multi_class='ovr', 
                    average='macro'
                )
        except:
            pass
        
        return results
    
    def compute_per_question_type_accuracy(self) -> Dict[str, float]:
        """Compute accuracy per question type."""
        if len(self.all_question_types) == 0:
            return {}
        
        results = {}
        for qt in self.question_types:
            mask = [i for i, q_type in enumerate(self.all_question_types) if q_type == qt]
            if len(mask) > 0:
                preds = [self.all_predictions[i] for i in mask]
                gt = [self.all_ground_truth[i] for i in mask]
                results[f'{qt}_accuracy'] = accuracy_score(gt, preds)
                results[f'{qt}_count'] = len(mask)
        
        return results
    
    def compute_confusion_matrix(self, top_k: int = 20):
        """
        Compute confusion matrix for top-k most frequent answers.
        
        Args:
            top_k: Number of top answers to include
        """
        predictions = np.array(self.all_predictions)
        ground_truth = np.array(self.all_ground_truth)
        
        # Get top-k most frequent answers
        unique, counts = np.unique(ground_truth, return_counts=True)
        top_k_indices = unique[np.argsort(-counts)[:top_k]]
        
        # Filter predictions and ground truth
        mask = np.isin(ground_truth, top_k_indices)
        filtered_preds = predictions[mask]
        filtered_gt = ground_truth[mask]
        
        # Compute confusion matrix
        cm = confusion_matrix(filtered_gt, filtered_preds, labels=top_k_indices)
        
        return cm, top_k_indices
    
    def get_detailed_results(self) -> pd.DataFrame:
        """Get detailed results for each prediction."""
        results = []
        for i in range(len(self.all_predictions)):
            pred_idx = self.all_predictions[i]
            gt_idx = self.all_ground_truth[i]
            
            result = {
                'prediction': self.idx_to_answer.get(pred_idx, 'unknown'),
                'ground_truth': self.idx_to_answer.get(gt_idx, 'unknown'),
                'correct': pred_idx == gt_idx,
                'pred_idx': pred_idx,
                'gt_idx': gt_idx
            }
            
            if len(self.all_question_types) > i:
                result['question_type'] = self.all_question_types[i]
            
            if len(self.all_questions) > i:
                result['question'] = self.all_questions[i]
            
            if len(self.all_probabilities) > i:
                result['confidence'] = self.all_probabilities[i][pred_idx]
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics at once."""
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = self.compute_accuracy()
        
        # Binary metrics
        metrics.update(self.compute_binary_metrics())
        
        # Multi-class metrics
        metrics.update(self.compute_multiclass_metrics())
        
        # AUC-ROC
        metrics.update(self.compute_auc_roc())
        
        # Per question type
        metrics.update(self.compute_per_question_type_accuracy())
        
        return metrics
    
    def print_summary(self):
        """Print evaluation summary."""
        metrics = self.compute_all_metrics()
        
        print("=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal samples: {len(self.all_predictions)}")
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        # Binary metrics
        if metrics.get('binary_accuracy', 0) > 0:
            print("\n--- Binary Questions (Yes/No) ---")
            print(f"Accuracy:  {metrics['binary_accuracy']:.4f}")
            print(f"Precision: {metrics['binary_precision']:.4f}")
            print(f"Recall:    {metrics['binary_recall']:.4f}")
            print(f"F1-Score:  {metrics['binary_f1']:.4f}")
            if 'binary_auc_roc' in metrics:
                print(f"AUC-ROC:   {metrics['binary_auc_roc']:.4f}")
        
        # Multi-class metrics
        print("\n--- Multi-class Metrics ---")
        print(f"Macro Precision:    {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:       {metrics['macro_recall']:.4f}")
        print(f"Macro F1-Score:     {metrics['macro_f1']:.4f}")
        if 'macro_auc_roc' in metrics:
            print(f"Macro AUC-ROC:      {metrics['macro_auc_roc']:.4f}")
        
        # Per question type
        print("\n--- Per Question Type Accuracy ---")
        for qt in self.question_types:
            key = f'{qt}_accuracy'
            count_key = f'{qt}_count'
            if key in metrics:
                print(f"{qt.capitalize():15s}: {metrics[key]:.4f} ({metrics[key]*100:.2f}%) "
                      f"[n={metrics.get(count_key, 0)}]")
        
        print("=" * 80)
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None, top_k: int = 15):
        """Plot confusion matrix for top-k answers."""
        cm, top_k_indices = self.compute_confusion_matrix(top_k)
        
        # Get answer labels
        labels = [self.idx_to_answer.get(idx, f'ans_{idx}') for idx in top_k_indices]
        
        # Create figure
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix (Top {top_k} Answers)', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Ground Truth', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        else:
            plt.show()
    
    def plot_per_type_accuracy(self, save_path: Optional[str] = None):
        """Plot accuracy per question type."""
        metrics = self.compute_per_question_type_accuracy()
        
        # Extract accuracies
        question_types = []
        accuracies = []
        counts = []
        
        for qt in self.question_types:
            key = f'{qt}_accuracy'
            count_key = f'{qt}_count'
            if key in metrics:
                question_types.append(qt.capitalize())
                accuracies.append(metrics[key] * 100)
                counts.append(metrics.get(count_key, 0))
        
        if len(question_types) == 0:
            print("No question type data available")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(question_types, accuracies, color='steelblue', alpha=0.8)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_xlabel('Question Type', fontsize=12)
        ax.set_title('Accuracy by Question Type', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-type accuracy plot saved to: {save_path}")
        else:
            plt.show()
    
    def save_results(self, save_path: str):
        """Save detailed results to CSV."""
        df = self.get_detailed_results()
        df.to_csv(save_path, index=False)
        print(f"Detailed results saved to: {save_path}")


def evaluate_model(model, dataloader, device, answer_vocab):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The VQA model
        dataloader: DataLoader for the dataset
        device: Device to run evaluation on
        answer_vocab: Answer vocabulary dictionary
    
    Returns:
        VQAEvaluator with all results
    """
    model.eval()
    evaluator = VQAEvaluator(answer_vocab)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            images = batch['image'].to(device)
            question_ids = batch['question']['input_ids'].to(device)
            question_mask = batch['question']['attention_mask'].to(device)
            answers = batch['answer_idx'].to(device)
            
            # Forward pass
            logits = model(images, question_ids, question_mask)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Add to evaluator
            evaluator.add_batch(
                predictions=predictions,
                ground_truth=answers,
                probabilities=probabilities,
                questions=batch['question']['text'],
                answers=batch.get('answer_normalized') or batch['answer']['text'],
                question_types=batch.get('question_type')
            )
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    print("VQA Evaluation Metrics Module")
    print("This module provides comprehensive evaluation for VQA models")
    print("\nFeatures:")
    print("  - Overall accuracy")
    print("  - Binary metrics (precision, recall, F1)")
    print("  - Multi-class metrics")
    print("  - AUC-ROC scores")
    print("  - Per-question-type analysis")
    print("  - Confusion matrices")
    print("  - Detailed result export")
