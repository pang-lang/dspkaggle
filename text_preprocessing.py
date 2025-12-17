#!/usr/bin/env python3
"""
Text Preprocessing Module for Radiology VQA
Handles text preprocessing for questions and answers in medical VQA tasks.
"""

import re
import string
import nltk
import spacy
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import torch

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class MedicalTextPreprocessor:
    """Comprehensive text preprocessing for medical VQA."""
    
    def __init__(self, 
                 model_name: str = "distilbert-base-uncased",
                 max_length: int = 64,
                 use_medical_terms: bool = True):
        """
        Initialize text preprocessor.
        
        Args:
            model_name: Name of the tokenizer model
            max_length: Maximum sequence length
            use_medical_terms: Whether to use medical terminology processing
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_medical_terms = use_medical_terms
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        
        # Initialize spaCy model for medical text
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Medical terminology patterns
        self.medical_patterns = {
            'anatomy': r'\b(heart|lung|liver|kidney|brain|spine|chest|abdomen|pelvis|skull|bone|muscle|tissue|organ)\b',
            'conditions': r'\b(cancer|tumor|lesion|mass|nodule|cyst|fracture|injury|disease|disorder|syndrome)\b',
            'procedures': r'\b(surgery|biopsy|scan|imaging|x-ray|mri|ct|ultrasound|examination|diagnosis)\b',
            'descriptors': r'\b(large|small|normal|abnormal|clear|opaque|dense|bright|dark|sharp|blurry)\b'
        }
        
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical punctuation
        text = re.sub(r'[^\w\s\-\.\,\?\!]', '', text)
        
        # Clean up punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'\!{2,}', '!', text)
        
        return text.strip()
    
    def extract_medical_terms(self, text: str) -> Dict[str, List[str]]:
        """Extract medical terms from text."""
        medical_terms = {}
        
        for category, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_terms[category] = list(set(matches))
        
        return medical_terms
    
    def preprocess_question(self, question: str) -> Dict[str, Union[str, List[str], Dict]]:
        """Preprocess a question text."""
        # Clean text
        cleaned = self.clean_text(question)
        
        processed_text = cleaned
        
        # Extract medical terms
        medical_terms = self.extract_medical_terms(processed_text)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(processed_text)
        
        # Get attention mask and input IDs without padding
        encoding = self.tokenizer(
            processed_text,
            max_length=self.max_length,
            padding=False,
            truncation=True
        )
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'original': question,
            'cleaned': cleaned,
            'expanded': processed_text,
            'tokens': tokens,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'medical_terms': medical_terms,
            'length': len(tokens)
        }
    
    def preprocess_answer(self, answer: str) -> Dict[str, Union[str, List[str], Dict]]:
        """Preprocess an answer text."""
        # Clean text
        cleaned = self.clean_text(answer)
        
        processed_text = cleaned
        
        # Extract medical terms
        medical_terms = self.extract_medical_terms(processed_text)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(processed_text)
        
        # Get attention mask and input IDs without padding
        encoding = self.tokenizer(
            processed_text,
            max_length=self.max_length,
            padding=False,
            truncation=True
        )
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'original': answer,
            'cleaned': cleaned,
            'expanded': processed_text,
            'tokens': tokens,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'medical_terms': medical_terms,
            'length': len(tokens)
        }
    
    def preprocess_qa_pair(self, question: str, answer: str) -> Dict[str, Dict]:
        """Preprocess a question-answer pair."""
        return {
            'question': self.preprocess_question(question),
            'answer': self.preprocess_answer(answer)
        }
    
    def batch_preprocess_texts(self, texts: List[str], text_type: str = 'question') -> List[Dict]:
        """Preprocess a batch of texts."""
        processed_texts = []
        
        for text in texts:
            if text_type == 'question':
                processed = self.preprocess_question(text)
            else:
                processed = self.preprocess_answer(text)
            processed_texts.append(processed)
        
        return processed_texts
    
    def analyze_text_statistics(self, texts: List[str]) -> Dict:
        """Analyze text statistics."""
        lengths = [len(text.split()) for text in texts]
        
        stats = {
            'total_texts': len(texts),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'std_length': np.std(lengths),
            'median_length': np.median(lengths)
        }
        
        return stats
    
    def create_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Create vocabulary from texts."""
        all_tokens = []
        
        for text in texts:
            processed = self.clean_text(text)
            tokens = self.tokenizer.tokenize(processed)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Create vocabulary with frequency
        vocabulary = dict(token_counts.most_common())
        
        return vocabulary
    
    def visualize_text_lengths(self, texts: List[str], title: str = "Text Length Distribution"):
        """Visualize text length distribution."""
        lengths = [len(text.split()) for text in texts]
        
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Text Length (words)')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return lengths
    
    def visualize_medical_terms(self, texts: List[str]):
        """Visualize medical terms distribution."""
        all_medical_terms = {'anatomy': [], 'conditions': [], 'procedures': [], 'descriptors': []}
        
        for text in texts:
            medical_terms = self.extract_medical_terms(text)
            for category, terms in medical_terms.items():
                all_medical_terms[category].extend(terms)
        
        # Count terms
        term_counts = {category: Counter(terms) for category, terms in all_medical_terms.items()}
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (category, counts) in enumerate(term_counts.items()):
            if counts:
                top_terms = counts.most_common(10)
                terms, counts_list = zip(*top_terms)
                
                axes[i].bar(range(len(terms)), counts_list)
                axes[i].set_title(f'{category.title()} Terms')
                axes[i].set_xlabel('Terms')
                axes[i].set_ylabel('Frequency')
                axes[i].set_xticks(range(len(terms)))
                axes[i].set_xticklabels(terms, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
        
        return term_counts

def create_medical_glossary() -> Dict[str, str]:
    """Create a medical glossary for reference."""
    glossary = {
        # Anatomy
        'heart': 'The muscular organ that pumps blood through the circulatory system',
        'lung': 'The respiratory organ where gas exchange occurs',
        'liver': 'The largest internal organ, responsible for detoxification and metabolism',
        'kidney': 'The organ that filters blood and produces urine',
        'brain': 'The central nervous system organ responsible for cognition',
        'spine': 'The vertebral column that supports the body',
        'chest': 'The upper part of the torso containing the heart and lungs',
        'abdomen': 'The part of the body between the chest and pelvis',
        
        # Conditions
        'cancer': 'A disease characterized by uncontrolled cell growth',
        'tumor': 'An abnormal mass of tissue',
        'lesion': 'An area of abnormal tissue',
        'mass': 'An abnormal collection of tissue',
        'nodule': 'A small, round mass of tissue',
        'cyst': 'A fluid-filled sac',
        'fracture': 'A break in a bone',
        
        # Procedures
        'surgery': 'A medical procedure involving cutting into the body',
        'biopsy': 'The removal of tissue for examination',
        'scan': 'An imaging procedure to visualize internal structures',
        'imaging': 'The process of creating images of internal structures',
        'x-ray': 'A form of electromagnetic radiation used for imaging',
        'mri': 'Magnetic resonance imaging',
        'ct': 'Computed tomography',
        'ultrasound': 'Imaging using sound waves',
        
        # Descriptors
        'opaque': 'Not allowing light to pass through',
        'dense': 'Having high density',
        'clear': 'Transparent or unobstructed',
        'abnormal': 'Not normal or typical',
        'normal': 'Within normal limits',
        'sharp': 'Well-defined or clear',
        'blurry': 'Not clear or well-defined'
    }
    
    return glossary

def test_text_preprocessing():
    """Test the text preprocessing functions."""
    print("=== Testing Text Preprocessing ===")
    
    # Sample medical texts
    sample_questions = [
        "What is the size of the mass in the right lung?",
        "Is there any evidence of cancer in this CT scan?",
        "What does the MRI show in the brain?",
        "Are there any fractures visible in the x-ray?",
        "What is the density of the lesion in the liver?"
    ]
    
    sample_answers = [
        "The mass measures 2.5 cm in diameter",
        "No evidence of malignancy is seen",
        "The MRI shows normal brain anatomy",
        "No fractures are identified",
        "The lesion appears hyperdense on CT"
    ]
    
    # Initialize preprocessor
    preprocessor = MedicalTextPreprocessor()
    
    # Test question preprocessing
    print("\n=== Question Preprocessing ===")
    for i, question in enumerate(sample_questions):
        processed = preprocessor.preprocess_question(question)
        print(f"Q{i+1}: {question}")
        print(f"  Cleaned: {processed['cleaned']}")
        print(f"  Medical terms: {processed['medical_terms']}")
        print(f"  Length: {processed['length']} tokens\n")
    
    # Test answer preprocessing
    print("\n=== Answer Preprocessing ===")
    for i, answer in enumerate(sample_answers):
        processed = preprocessor.preprocess_answer(answer)
        print(f"A{i+1}: {answer}")
        print(f"  Cleaned: {processed['cleaned']}")
        print(f"  Medical terms: {processed['medical_terms']}")
        print(f"  Length: {processed['length']} tokens\n")
    
    # Test batch processing
    print("\n=== Batch Processing ===")
    batch_processed = preprocessor.batch_preprocess_texts(sample_questions, 'question')
    print(f"Processed {len(batch_processed)} questions")
    
    # Analyze statistics
    print("\n=== Text Statistics ===")
    stats = preprocessor.analyze_text_statistics(sample_questions)
    print(f"Average length: {stats['avg_length']:.2f} words")
    print(f"Length range: {stats['min_length']} - {stats['max_length']} words")
    
    # Create vocabulary
    print("\n=== Vocabulary ===")
    vocab = preprocessor.create_vocabulary(sample_questions + sample_answers)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Top 10 tokens: {list(vocab.keys())[:10]}")
    
    return preprocessor, batch_processed

if __name__ == "__main__":
    # Test the preprocessing
    preprocessor, results = test_text_preprocessing()
    print("Text preprocessing test completed!")
