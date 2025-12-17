#!/usr/bin/env python3

import os
from collections import Counter

from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np

EDA_OUTPUT_DIR = "eda_reports"
BINARY_ANSWERS = {"yes", "no"}
UNK_ANSWER_TOKEN = "<unk>"
ANSWER_SYNONYMS = {
    'no abnormality': 'no',
    'no abnormalities': 'no',
    'no abnormality detected': 'no',
    'no evidence of disease': 'no',
    'none': 'no',
    'not seen': 'no',
    'absent': 'no',
    'without': 'no',
    'normal study': 'normal',
    'unremarkable': 'normal',
    'female patient': 'female',
    'woman': 'female',
    'male patient': 'male',
    'man': 'male'
}


def normalize_answer(answer: str) -> str:
    """Normalize answer for vocabulary statistics."""
    if not isinstance(answer, str):
        return ""
    normalized = answer.lower().strip()
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch.isspace())
    normalized = " ".join(normalized.split())
    if normalized in ANSWER_SYNONYMS:
        normalized = ANSWER_SYNONYMS[normalized]
    return normalized


def derive_question_type(answer: str) -> str:
    """Return 'binary' for yes/no answers, otherwise 'open-ended'."""
    normalized = normalize_answer(answer)
    return 'binary' if normalized in BINARY_ANSWERS else 'open-ended'


# Hugging face dataset 
def load_vqa_rad_dataset():
    print("Loading vqa-rad dataset...")
    ds = load_dataset("flaviagiammarino/vqa-rad")
    return ds

def explore_dataset(ds):
    """Explore the dataset structure and content."""
    print("Dataset structure:")
    print(f"Keys: {list(ds.keys())}")
    
    for split in ds.keys():
        print(f"\n{split} split:")
        print(f"  Number of samples: {len(ds[split])}")
        print(f"  Features: {ds[split].features}")
        
        # Show first few examples
        print(f"\nFirst 3 examples from {split}:")
        for i in range(min(3, len(ds[split]))):
            example = ds[split][i]
            print(f"  Example {i+1}:")
            for key, value in example.items():
                if key == 'image':
                    print(f"    {key}: PIL Image {value.size}")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")

def analyze_questions(ds, output_dir: str = EDA_OUTPUT_DIR):
    """Analyze the questions in the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    all_questions = []
    all_answers = []
    
    for split in ds.keys():
        questions = ds[split]['question']
        answers = ds[split]['answer']
        all_questions.extend(questions)
        all_answers.extend(answers)
    
    print(f"\nDataset Analysis:")
    print(f"Total questions: {len(all_questions)}")
    print(f"Total answers: {len(all_answers)}")
    
    # Question length analysis
    question_lengths = [len(q.split()) for q in all_questions]
    print(f"Average question length: {np.mean(question_lengths):.2f} words")
    print(f"Question length range: {min(question_lengths)} - {max(question_lengths)} words")
    length_df = pd.DataFrame({'question_length': question_lengths})
    length_df.to_csv(os.path.join(output_dir, 'question_length_distribution.csv'), index=False)
    
    plt.figure(figsize=(8, 4))
    sns.histplot(question_lengths, bins=30, kde=False)
    plt.xlabel('Question length (words)')
    plt.ylabel('Frequency')
    plt.title('Question Length Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'question_length_hist.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    # Answer analysis
    unique_answers = set(all_answers)
    print(f"Unique answers: {len(unique_answers)}")
    
    answer_counts = Counter(all_answers)
    top_answers = answer_counts.most_common(20)
    print(f"\nTop 10 most common answers:")
    for answer, count in top_answers[:10]:
        print(f"  '{answer}': {count} times")
    pd.DataFrame(top_answers, columns=['answer', 'count']).to_csv(
        os.path.join(output_dir, 'top_answer_counts.csv'), index=False
    )
    
    binary_counts = {ans: answer_counts.get(ans, 0) for ans in ['yes', 'no']}
    total_binary = sum(binary_counts.values())
    if total_binary:
        yes_ratio = binary_counts['yes'] / total_binary
        print(f"\nBinary answers found: {total_binary} (Yes ratio: {yes_ratio:.2%})")
        pd.Series(binary_counts).to_csv(os.path.join(output_dir, 'binary_answer_counts.csv'))
    else:
        print("\nNo explicit yes/no answers detected.")

def analyze_question_types(ds, output_dir: str = EDA_OUTPUT_DIR):
    """Summarize derived binary vs open-ended question distribution."""
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    for split in ds.keys():
        answers = ds[split]['answer']
        derived = [derive_question_type(ans) for ans in answers]
        counts = Counter(derived)
        if not counts:
            continue
        df = pd.DataFrame(
            {'question_type': list(counts.keys()), 'count': list(counts.values())}
        ).sort_values('count', ascending=False)
        df['split'] = split
        df.to_csv(os.path.join(output_dir, f'question_type_counts_{split}.csv'), index=False)
        frames.append(df)
    
    if frames:
        combined = pd.concat(frames)
        plt.figure(figsize=(6, 4))
        sns.barplot(data=combined, x='question_type', y='count', hue='split')
        plt.ylabel('Count')
        plt.xlabel('Derived Question Type')
        plt.title('Binary vs Open-ended Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'question_type_distribution.png'), dpi=200, bbox_inches='tight')
        plt.close()

def analyze_answer_coverage(ds, output_dir: str = EDA_OUTPUT_DIR):
    """Plot cumulative coverage as a function of K most frequent answers."""
    os.makedirs(output_dir, exist_ok=True)
    all_answers = []
    for split in ds.keys():
        all_answers.extend(ds[split]['answer'])
    
    counts = Counter()
    for ans in all_answers:
        normalized = normalize_answer(ans) or UNK_ANSWER_TOKEN
        counts[normalized] += 1
    
    total = sum(counts.values()) or 1
    sorted_counts = counts.most_common()
    cumulative = 0
    coverage_rows = []
    for k, (answer, count) in enumerate(sorted_counts, start=1):
        cumulative += count
        coverage = cumulative / total
        coverage_rows.append({
            'k': k,
            'answer': answer,
            'count': count,
            'coverage': coverage
        })
    
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(os.path.join(output_dir, 'answer_coverage_curve.csv'), index=False)
    
    plt.figure(figsize=(8, 4))
    plt.plot(coverage_df['k'], coverage_df['coverage'], label='Coverage')
    plt.axhline(0.95, color='red', linestyle='--', label='95% threshold')
    plt.xlabel('Top-K answers')
    plt.ylabel('Cumulative coverage')
    plt.title('Answer Coverage vs Top-K')
    plt.ylim(0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'answer_coverage_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()

def visualize_dataset(ds, num_samples=5, output_dir: str = EDA_OUTPUT_DIR):
    """Visualize sample images and questions."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nVisualizing {num_samples} samples...")
    
    # Get samples from the first available split
    split_name = list(ds.keys())[0]
    samples = ds[split_name].select(range(min(num_samples, len(ds[split_name]))))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i, sample in enumerate(samples):
        # Display image
        image = sample['image']
        axes[i].imshow(image)
        axes[i].set_title(f"Q: {sample['question'][:50]}...\nA: {sample['answer']}", 
                         fontsize=8, wrap=True)
        axes[i].axis('off')
    
    plt.tight_layout()
    figure_path = os.path.join(output_dir, 'sample_images.png')
    plt.savefig(figure_path, dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to load and explore the dataset."""
    try:
        ds = load_vqa_rad_dataset()
        explore_dataset(ds)
        os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
        analyze_questions(ds, output_dir=EDA_OUTPUT_DIR)
        analyze_question_types(ds, output_dir=EDA_OUTPUT_DIR)
        analyze_answer_coverage(ds, output_dir=EDA_OUTPUT_DIR)
        visualize_dataset(ds, output_dir=EDA_OUTPUT_DIR)
        
        print("\nDataset loaded successfully!")
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    dataset = main()
