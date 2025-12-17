#!/usr/bin/env python3
"""
Combined Image and Text Preprocessing for Radiology VQA
Integrates image and text preprocessing for complete VQA data preparation.
"""

import json
import os
import string

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Tuple, Optional, Union

from image_preprocessing import (
    MedicalImagePreprocessor,
    get_or_compute_dataset_stats
)
from text_preprocessing import MedicalTextPreprocessor
import matplotlib.pyplot as plt
from collections import Counter

UNK_ANSWER_TOKEN = "<unk>"
BINARY_ANSWERS = {
    "yes",
    "no",
    "normal",
    "abnormal",
    "no abnormality",
    "no acute abnormality",
    "no acute findings",
    "unremarkable",
    "negative",
    "positive",
}
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_PATH = os.path.join('data_splits', f'vqa_rad_seed{DEFAULT_SPLIT_SEED}.json')
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
    'man': 'male',
    'cardiac enlargement': 'cardiomegaly',
    'enlarged cardiac silhouette': 'cardiomegaly',
    'enlarged heart': 'cardiomegaly',
    'cardiac silhouette enlargement': 'cardiomegaly',
    'pulmonary edema': 'edema',
    'pulmonary congestion': 'edema',
    'fluid in the lungs': 'edema',
    'pleural effusion': 'effusion',
    'pericardial effusion': 'effusion',
    'fluid in pleural space': 'effusion',
    'brain bleed': 'hemorrhage',
    'intracranial hemorrhage': 'hemorrhage',
    'ich': 'hemorrhage',
    'fractured': 'fracture',
    'broken bone': 'fracture',
    'collapsed lung': 'pneumothorax',
    'lung collapse': 'pneumothorax',
    'air in pleural space': 'pneumothorax'
}


def normalize_answer_text(answer: str) -> str:
    """Lowercase, strip punctuation/whitespace, and unify synonyms."""
    if not isinstance(answer, str):
        return UNK_ANSWER_TOKEN
    text = answer.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())  # collapse whitespace
    if not text:
        return UNK_ANSWER_TOKEN
    if text in ANSWER_SYNONYMS:
        text = ANSWER_SYNONYMS[text]
    return text or UNK_ANSWER_TOKEN


def derive_question_type(answer: str) -> str:
    """Return 'binary' when the normalized answer is yes/no, else open-ended."""
    normalized = normalize_answer_text(answer)
    return 'binary' if normalized in BINARY_ANSWERS else 'open-ended'


def get_or_create_split_indices(dataset_name: str,
                                base_dataset=None,
                                split_path: str = DEFAULT_SPLIT_PATH,
                                seed: int = DEFAULT_SPLIT_SEED,
                                use_official_test_split: bool = True) -> Dict[str, List[int]]:
    """
    Create (or load) dataset split indices.
    
    If use_official_test_split=True (default):
        - Uses official HuggingFace train (1,793) and test (451) splits
        - Splits train into train (80%) and val (20%)
        - Total: 1,434 train / 359 val / 451 test = 2,244 samples
    
    If use_official_test_split=False (legacy):
        - Uses only HuggingFace train split (1,793)
        - Splits into train (70%) / val (15%) / test (15%)
        - Total: 1,255 train / 269 val / 269 test = 1,793 samples
    """
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    if os.path.exists(split_path):
        with open(split_path, 'r') as fp:
            stored = json.load(fp)
        # Convert list values, preserve other metadata
        result = {}
        for k, v in stored.items():
            if isinstance(v, list):
                result[k] = list(v)
            else:
                result[k] = v  # Preserve metadata like use_official_test, train_size, etc.
        return result
    
    if use_official_test_split:
        # Load both train and test from HuggingFace
        full_dataset = load_dataset(dataset_name)
        train_dataset = full_dataset['train']  # 1,793 samples
        test_dataset = full_dataset['test']    # 451 samples
        
        # Split train into train/val (80/20)
        labels = [derive_question_type(train_dataset[i]['answer']) for i in range(len(train_dataset))]
        indices = np.arange(len(train_dataset))
        
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=seed,
            stratify=labels
        )
        
        # Test indices are offset by len(train_dataset) to indicate they come from test split
        # We'll store them as: [len(train), len(train)+1, ..., len(train)+len(test)-1]
        test_idx = np.arange(len(train_dataset), len(train_dataset) + len(test_dataset))
        
        splits = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
            'use_official_test': True,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset)
        }
    else:
        # Legacy: use only train split, create custom 70/15/15 split
        if base_dataset is None:
            base_dataset = load_dataset(dataset_name)['train']
        
        labels = [derive_question_type(base_dataset[i]['answer']) for i in range(len(base_dataset))]
        indices = np.arange(len(base_dataset))
        
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices,
            labels,
            train_size=0.7,
            random_state=seed,
            stratify=labels
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=0.5,
            random_state=seed,
            stratify=[labels[i] for i in temp_idx]
        )
        
        splits = {
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
            'use_official_test': False
        }
    
    with open(split_path, 'w') as fp:
        json.dump(splits, fp)
    print(f"✓ Created new splits: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
    return splits



class CombinedVQADataset(Dataset):
    """Combined dataset for image and text preprocessing."""
    
    def __init__(self, 
                 dataset_name: str = "flaviagiammarino/vqa-rad",
                 split: str = "train",
                 image_preprocessor: Optional[MedicalImagePreprocessor] = None,
                 text_preprocessor: Optional[MedicalTextPreprocessor] = None,
                 max_samples: Optional[int] = None,
                 question_types: Optional[List[str]] = None,
                 answer_vocab: Optional[Dict[str, int]] = None,
                 allow_vocab_updates: bool = False,
                 max_answer_vocab_size: Optional[int] = None,
                 preprocess_steps: Optional[List[str]] = None,
                 encode_answers: bool = False,
                 coverage_threshold: float = 0.95,
                 base_dataset=None,
                 indices: Optional[List[int]] = None):
        """
        Initialize combined dataset.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            split: Dataset split (for logging only when custom indices are provided)
            image_preprocessor: Image preprocessor instance
            text_preprocessor: Text preprocessor instance
            max_samples: Maximum number of samples to load
            question_types: Filter by derived question types ('binary' or 'open-ended')
            answer_vocab: Shared answer vocabulary across splits
            allow_vocab_updates: Whether new answers can extend the vocabulary
            max_answer_vocab_size: Maximum number of answers to keep (most frequent)
            preprocess_steps: Optional image preprocessing steps applied before PyTorch transforms
            encode_answers: Whether to run full tokenizer encoding on answers (disabled by default)
            coverage_threshold: Minimum cumulative coverage to reach when building vocab
            base_dataset: Optional preloaded HuggingFace dataset to avoid repeated downloads
            indices: Optional list of dataset indices to select (for deterministic splits)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.image_preprocessor = image_preprocessor or MedicalImagePreprocessor()
        self.text_preprocessor = text_preprocessor or MedicalTextPreprocessor()
        self.max_samples = max_samples
        self.question_types = question_types
        self.max_answer_vocab_size = max_answer_vocab_size
        self.preprocess_steps = preprocess_steps or []
        self.encode_answers = encode_answers
        self.coverage_threshold = coverage_threshold
        self.coverage_curve: List[Dict[str, float]] = []
        self.allow_vocab_updates = allow_vocab_updates
        
        # Load dataset once and store selected indices
        if base_dataset is not None:
            self.dataset = base_dataset
        else:
            self.dataset = load_dataset(dataset_name)['train']
        
        if indices is not None:
            self.indices = list(indices)
        else:
            self.indices = list(range(len(self.dataset)))
        
        if question_types:
            target_types = set(question_types)
            self.indices = [
                idx for idx in self.indices
                if self._derive_question_type(self.dataset[idx].get('answer', '')) in target_types
            ]
        
        if max_samples:
            self.indices = self.indices[:min(max_samples, len(self.indices))]
        
        print(f"Loaded {len(self.indices)} indexed samples for {split} split")
        
        # Initialize / build answer vocabulary
        self.answer_vocab = answer_vocab if answer_vocab is not None else {}
        if self.allow_vocab_updates and not self.answer_vocab:
            self.answer_vocab[UNK_ANSWER_TOKEN] = 0
        elif UNK_ANSWER_TOKEN not in self.answer_vocab:
            # Ensure unknown token is always present for consistent indexing
            self.answer_vocab[UNK_ANSWER_TOKEN] = 0
        
        if allow_vocab_updates:
            self._populate_answer_vocab()
            # Prevent further growth so model size stays fixed
            self.allow_vocab_updates = False
       
        # Convenience reverse mapping (used by evaluation)
        self.idx_to_answer = {idx: ans for ans, idx in self.answer_vocab.items()}
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer text for stable vocabulary lookups."""
        cleaned = self.text_preprocessor.clean_text(answer) if self.text_preprocessor else answer
        normalized = normalize_answer_text(cleaned)
        return normalized if normalized else UNK_ANSWER_TOKEN
    
    def _populate_answer_vocab(self):
        """Populate vocabulary with answers present in this split."""
        counts = Counter()
        for idx in self.indices:
            sample = self.dataset[idx]
            normalized = self._normalize_answer(sample.get('answer', ''))
            counts[normalized] += 1
        
        total = sum(counts.values()) or 1
        sorted_answers = counts.most_common()
        
        kept_answers = []
        cumulative = 0
        coverage_curve = []
        limit = self.max_answer_vocab_size or len(sorted_answers)
        
        for rank, (answer, count) in enumerate(sorted_answers, start=1):
            if answer == UNK_ANSWER_TOKEN:
                continue
            cumulative += count
            coverage = cumulative / total
            coverage_curve.append({
                'k': rank,
                'answer': answer,
                'count': count,
                'coverage': coverage
            })
            kept_answers.append(answer)
            if coverage >= self.coverage_threshold or len(kept_answers) >= limit:
                break
        
        self.coverage_curve = coverage_curve
        
        # Ensure UNK token exists
        if UNK_ANSWER_TOKEN not in self.answer_vocab:
            self.answer_vocab[UNK_ANSWER_TOKEN] = 0
        
        next_index = max(self.answer_vocab.values()) + 1 if self.answer_vocab else 1
        for answer in kept_answers:
            if answer not in self.answer_vocab:
                self.answer_vocab[answer] = next_index
                next_index += 1
    
    def _derive_question_type(self, answer: str) -> str:
        """Return 'binary' if answer is yes/no else 'open-ended'."""
        normalized = self._normalize_answer(answer)
        return 'binary' if normalized in BINARY_ANSWERS else 'open-ended'
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample with combined preprocessing."""
        dataset_idx = self.indices[idx]
        sample = self.dataset[dataset_idx]
        
        # Get image
        image = sample['image']
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        image_tensor = self.image_preprocessor.prepare_image(
            image_np,
            is_training=(self.split == 'train'),
            preprocess_steps=self.preprocess_steps
        )
        
        # Get text data
        question = sample['question']
        answer = sample['answer']
        
        # Preprocess text
        if self.text_preprocessor:
            processed_question = self.text_preprocessor.preprocess_question(question)
        else:
            # Basic text processing
            processed_question = {
                'input_ids': torch.tensor([0]),
                'attention_mask': torch.tensor([0]),
                'tokens': [],
                'medical_terms': {}
            }
        
        if self.encode_answers and self.text_preprocessor:
            processed_answer = self.text_preprocessor.preprocess_answer(answer)
            answer_cleaned = processed_answer['cleaned']
            answer_medical_terms = processed_answer.get('medical_terms', {})
        else:
            answer_cleaned = self.text_preprocessor.clean_text(answer) if self.text_preprocessor else (answer or "")
            answer_medical_terms = self.text_preprocessor.extract_medical_terms(answer_cleaned) if self.text_preprocessor else {}
            processed_answer = {
                'cleaned': answer_cleaned,
                'medical_terms': answer_medical_terms
            }
        
        normalized_answer = self._normalize_answer(answer)
        question_type = 'binary' if normalized_answer in BINARY_ANSWERS else 'open-ended'
        if normalized_answer not in self.answer_vocab:
            if self.allow_vocab_updates:
                new_index = max(self.answer_vocab.values()) + 1
                self.answer_vocab[normalized_answer] = new_index
                self.idx_to_answer[new_index] = normalized_answer
            else:
                normalized_answer = UNK_ANSWER_TOKEN
        
        answer_index = self.answer_vocab.get(normalized_answer, self.answer_vocab.get(UNK_ANSWER_TOKEN, 0))
        
        answer_payload = {
            'text': answer,
            'cleaned': answer_cleaned,
            'medical_terms': answer_medical_terms,
            'tokens': processed_answer.get('tokens', [])
        }
        if self.encode_answers:
            answer_payload.update({
                'input_ids': processed_answer['input_ids'],
                'attention_mask': processed_answer['attention_mask'],
                'tokens': processed_answer.get('tokens', []),
                'pad_token_id': self.text_preprocessor.pad_token_id if self.text_preprocessor else 0
            })
        
        return {
            'image': image_tensor,
            'question': {
                'text': question,
                'input_ids': processed_question['input_ids'],
                'attention_mask': processed_question['attention_mask'],
                'tokens': processed_question.get('tokens', []),
                'medical_terms': processed_question.get('medical_terms', {}),
                'pad_token_id': self.text_preprocessor.pad_token_id if self.text_preprocessor else 0
            },
            'answer': answer_payload,
            'image_id': sample.get('image_id', idx),
            'question_id': sample.get('question_id', idx),
            'answer_idx': torch.tensor(answer_index, dtype=torch.long),
            'answer_normalized': normalized_answer,
            'question_type': question_type
        }

class CombinedVQADataModule:
    """Data module for combined image and text preprocessing."""
    
    def __init__(self, 
                 dataset_name: str = "flaviagiammarino/vqa-rad",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 max_samples: Optional[int] = None,
                 question_types: Optional[List[str]] = None,
                 image_target_size: Tuple[int, int] = (224, 224),
                 text_model_name: str = "distilbert-base-uncased",
                 text_max_length: int = 40,
                 max_answer_vocab_size: Optional[int] = 500,
                 coverage_threshold: float = 0.95,
                 image_preprocess_steps: Optional[List[str]] = None,
                 encode_answers: bool = False,
                 split_seed: int = DEFAULT_SPLIT_SEED,
                 split_path: str = DEFAULT_SPLIT_PATH,
                 stats_max_samples: Optional[int] = 2000):
        """
        Initialize combined data module.
        
        Args:
            dataset_name: Name of the Hugging Face dataset
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            max_samples: Maximum samples per split
            question_types: Filter by question types
            image_target_size: Target size for images
            text_model_name: Name of the text model
            text_max_length: Maximum text length
            image_preprocess_steps: Optional classical preprocessing applied before transforms
            encode_answers: Whether to tokenize answers (set False to save time)
            stats_max_samples: Max samples to use when estimating dataset-level normalization
        """
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.question_types = question_types
        self.max_answer_vocab_size = max_answer_vocab_size
        self.coverage_threshold = coverage_threshold
        self.image_preprocess_steps = image_preprocess_steps or []
        self.encode_answers = encode_answers
        self.split_seed = split_seed
        self.split_path = split_path
        self.stats_max_samples = stats_max_samples
        
        # Initialize preprocessors
        self.image_preprocessor = MedicalImagePreprocessor(target_size=image_target_size)
        stats = get_or_compute_dataset_stats(
            dataset_name=self.dataset_name,
            split='train',
            target_size=image_target_size,
            preprocess_steps=self.image_preprocess_steps,
            max_samples=self.stats_max_samples
        )
        if stats:
            self.image_preprocessor.set_normalization_stats(stats['mean'], stats['std'])
        self.text_preprocessor = MedicalTextPreprocessor(
            model_name=text_model_name,
            max_length=text_max_length
        )
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.answer_vocab: Dict[str, int] = {UNK_ANSWER_TOKEN: 0}
        self.base_dataset = None
        self.test_base_dataset = None
        self.split_indices: Optional[Dict[str, List[int]]] = None
    
    def setup(self, stage: str = None, use_official_test_split: bool = True):
        """Setup datasets for different stages."""
        # Initialize test_base_dataset attribute if not exists
        if not hasattr(self, 'test_base_dataset'):
            self.test_base_dataset = None
        
        # Load or create split indices first
        if self.split_indices is None:
            self.split_indices = get_or_create_split_indices(
                self.dataset_name,
                base_dataset=self.base_dataset,
                split_path=self.split_path,
                seed=self.split_seed,
                use_official_test_split=use_official_test_split
            )
        
        # Now load datasets based on split configuration
        if self.base_dataset is None:
            full_dataset = load_dataset(self.dataset_name)
            self.base_dataset = full_dataset['train']
            
            # Load test dataset only if using official split
            if self.split_indices.get('use_official_test', False):
                self.test_base_dataset = full_dataset['test']
        
        if stage == 'fit' or stage is None:
            self.train_dataset = CombinedVQADataset(
                dataset_name=self.dataset_name,
                split='train',
                image_preprocessor=self.image_preprocessor,
                text_preprocessor=self.text_preprocessor,
                max_samples=self.max_samples,
                question_types=self.question_types,
                answer_vocab=self.answer_vocab,
                allow_vocab_updates=True,
                max_answer_vocab_size=self.max_answer_vocab_size,
                preprocess_steps=self.image_preprocess_steps,
                encode_answers=self.encode_answers,
                coverage_threshold=self.coverage_threshold,
                base_dataset=self.base_dataset,
                indices=self.split_indices['train']
            )
            self.answer_vocab = self.train_dataset.answer_vocab
            
            # Use validation split if available, otherwise use part of train
            self.val_dataset = CombinedVQADataset(
                dataset_name=self.dataset_name,
                split='val',
                image_preprocessor=self.image_preprocessor,
                text_preprocessor=self.text_preprocessor,
                max_samples=self.max_samples,
                question_types=self.question_types,
                answer_vocab=self.answer_vocab,
                allow_vocab_updates=False,
                max_answer_vocab_size=self.max_answer_vocab_size,
                preprocess_steps=self.image_preprocess_steps,
                encode_answers=self.encode_answers,
                coverage_threshold=self.coverage_threshold,
                base_dataset=self.base_dataset,
                indices=self.split_indices['val']
            )
        
        if stage == 'test' or stage is None:
            # Determine which base dataset to use for test split
            use_official = self.split_indices.get('use_official_test', False)
            
            if use_official and self.test_base_dataset is not None:
                # Using official test split: use test_base_dataset with 0-based indices
                test_base = self.test_base_dataset
                test_indices = list(range(len(self.test_base_dataset)))
            else:
                # Using custom split from train data: use base_dataset with stored indices
                test_base = self.base_dataset
                test_indices = self.split_indices['test']
            
            self.test_dataset = CombinedVQADataset(
                dataset_name=self.dataset_name,
                split='test',
                image_preprocessor=self.image_preprocessor,
                text_preprocessor=self.text_preprocessor,
                max_samples=self.max_samples,
                question_types=self.question_types,
                answer_vocab=self.answer_vocab,
                allow_vocab_updates=False,
                max_answer_vocab_size=self.max_answer_vocab_size,
                preprocess_steps=self.image_preprocess_steps,
                encode_answers=self.encode_answers,
                coverage_threshold=self.coverage_threshold,
                base_dataset=test_base,
                indices=test_indices
            )
    
    def train_dataloader(self):
        """Create training data loader."""
        if self.train_dataset is None:
            self.setup('fit')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        """Create validation data loader."""
        if self.val_dataset is None:
            self.setup('fit')
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        """Create test data loader."""
        if self.test_dataset is None:
            self.setup('test')
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

def create_combined_data_loaders(dataset_name: str = "flaviagiammarino/vqa-rad",
                              batch_size: int = 32,
                              num_workers: int = 4,
                              max_samples: Optional[int] = None,
                              image_target_size: Tuple[int, int] = (224, 224),
                              text_model_name: str = "distilbert-base-uncased",
                              text_max_length: int = 40,
                              max_answer_vocab_size: Optional[int] = 500,
                              image_preprocess_steps: Optional[List[str]] = None,
                              coverage_threshold: float = 0.95,
                              encode_answers: bool = False,
                              split_seed: int = DEFAULT_SPLIT_SEED,
                              split_path: str = DEFAULT_SPLIT_PATH,
                              stats_max_samples: Optional[int] = 2000,
                              base_dataset=None,
                              use_official_test_split: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Dict[str, int], Dict[str, List[int]]]:
                           
    
    """
    Create combined data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        max_samples: Maximum samples per split
        image_target_size: Target size for images
        text_model_name: Name of the text model
        text_max_length: Maximum text length
        max_answer_vocab_size: Maximum number of answers to keep (most frequent)
        coverage_threshold: Minimum cumulative coverage for retained answers
        image_preprocess_steps: Optional classical preprocessing steps applied before transforms
        encode_answers: Whether to tokenize answer text (set False for lightweight training)
        split_seed: Random seed used for deterministic splits
        split_path: File path where split indices are stored
        stats_max_samples: Max samples to use when estimating dataset normalization stats
        use_official_test_split: If True, uses official HF train (1,793) and test (451) splits.
                                 If False, only uses HF train split and creates custom 70/15/15 split.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, answer_vocab, split_indices)
    """

    # Initialize data module
    data_module = CombinedVQADataModule(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        image_target_size=image_target_size,
        text_model_name=text_model_name,
        text_max_length=text_max_length,
        max_answer_vocab_size=max_answer_vocab_size,
        coverage_threshold=coverage_threshold,
        image_preprocess_steps=image_preprocess_steps,
        encode_answers=encode_answers,
        split_seed=split_seed,
        split_path=split_path,
        stats_max_samples=stats_max_samples
    )
    
    # Set base_dataset if provided (for backwards compatibility)
    if base_dataset is not None:
        data_module.base_dataset = base_dataset
    
    # Setup datasets - this will load both train and test if needed
    data_module.setup(use_official_test_split=use_official_test_split)
    
    # Create data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader() if data_module.test_dataset else None
    
    return train_loader, val_loader, test_loader, data_module.answer_vocab, data_module.split_indices

def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    question_pad_token_id = batch[0]['question'].get('pad_token_id', 0)
    answer_pad_token_id = batch[0]['answer'].get('pad_token_id', question_pad_token_id)
    # Separate different types of data
    images = []
    question_texts = []
    answer_texts = []
    question_input_ids = []
    question_attention_masks = []
    question_tokens = []
    question_medical_terms = []
    answer_tokens = []
    answer_medical_terms = []
    image_ids = []
    question_ids = []
    answer_indices = []
    answer_normalized = []
    answer_cleaned = []
    question_types = []
    answer_input_ids = []
    answer_attention_masks = []
    
    for item in batch:
        images.append(item['image'])
        question_texts.append(item['question']['text'])
        answer_texts.append(item['answer']['text'])
        question_input_ids.append(item['question']['input_ids'])
        question_attention_masks.append(item['question']['attention_mask'])
        question_tokens.append(item['question']['tokens'])
        question_medical_terms.append(item['question']['medical_terms'])
        answer_cleaned.append(item['answer'].get('cleaned', item['answer']['text']))
        answer_medical_terms.append(item['answer'].get('medical_terms', {}))
        if 'tokens' in item['answer']:
            answer_tokens.append(item['answer']['tokens'])
        else:
            answer_tokens.append([])
        if 'input_ids' in item['answer']:
            answer_input_ids.append(item['answer']['input_ids'])
            answer_attention_masks.append(item['answer']['attention_mask'])
        image_ids.append(item['image_id'])
        question_ids.append(item['question_id'])
        answer_indices.append(item['answer_idx'])
        answer_normalized.append(item['answer_normalized'])
        question_types.append(item.get('question_type', 'unknown'))
    
    # Stack tensors
    images = torch.stack(images)
    question_input_ids = pad_sequence(question_input_ids, batch_first=True, padding_value=question_pad_token_id)
    question_attention_masks = pad_sequence(question_attention_masks, batch_first=True, padding_value=0)
    answer_indices = torch.stack(answer_indices)
    
    answer_bundle = {
        'text': answer_texts,
        'cleaned': answer_cleaned,
        'medical_terms': answer_medical_terms,
        'tokens': answer_tokens
    }
    
    if answer_input_ids:
        answer_bundle['input_ids'] = pad_sequence(
            answer_input_ids, batch_first=True, padding_value=answer_pad_token_id
        )
        answer_bundle['attention_mask'] = pad_sequence(
            answer_attention_masks, batch_first=True, padding_value=0
        )
        answer_bundle['pad_token_id'] = answer_pad_token_id
    
    return {
        'image': images,
        'question': {
            'text': question_texts,
            'input_ids': question_input_ids,
            'attention_mask': question_attention_masks,
            'tokens': question_tokens,
            'medical_terms': question_medical_terms,
            'pad_token_id': question_pad_token_id
        },
        'answer': answer_bundle,
        'image_id': torch.tensor(image_ids),
        'question_id': torch.tensor(question_ids),
        'answer_idx': answer_indices,
        'answer_normalized': answer_normalized,
        'question_type': question_types
    }

def visualize_combined_batch(batch: Dict, num_samples: int = 4):
    """Visualize a batch of combined data."""
    images = batch['image']
    questions = batch['question']
    answers = batch['answer']
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i] * std.view(3, 1, 1) + mean.view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        # Display image
        axes[i].imshow(img.permute(1, 2, 0))
        
        # Get text data
        question_text = questions['text'][i]
        answer_text = answers['text'][i]
        
        axes[i].set_title(f"Q: {question_text[:50]}...\nA: {answer_text}", 
                         fontsize=8, wrap=True)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def compare_preprocessing_sample(dataset: CombinedVQADataset,
                                 sample_idx: int = 0,
                                 preprocess_steps: Optional[List[str]] = None,
                                 is_training: bool = True):
    """Visualize raw vs processed image for a sample index."""
    raw_sample = dataset.dataset[dataset.indices[sample_idx]]
    raw_image = np.array(raw_sample['image'])
    
    processed_tensor = dataset.image_preprocessor.prepare_image(
        raw_image,
        is_training=is_training,
        preprocess_steps=preprocess_steps or dataset.preprocess_steps
    )
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    processed_np = processed_tensor.cpu() * std + mean
    processed_np = processed_np.permute(1, 2, 0).numpy()
    processed_np = np.clip(processed_np, 0, 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(processed_np)
    axes[1].set_title('Processed')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def compare_combined_sample(dataset: CombinedVQADataset,
                            sample_idx: int = 0,
                            preprocess_steps: Optional[List[str]] = None,
                            is_training: bool = True):
    """Compare raw vs fully processed sample (image + text)."""
    raw_idx = dataset.indices[sample_idx]
    raw_sample = dataset.dataset[raw_idx]
    processed_sample = dataset[sample_idx]
    
    # Image comparison
    raw_image = np.array(raw_sample['image'])
    processed_tensor = processed_sample['image']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    processed_np = processed_tensor.cpu() * std + mean
    processed_np = processed_np.permute(1, 2, 0).numpy()
    processed_np = np.clip(processed_np, 0, 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(processed_np)
    axes[1].set_title('Processed Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    # Text comparison
    if dataset.text_preprocessor:
        question_processed = dataset.text_preprocessor.preprocess_question(raw_sample['question'])
        answer_processed = dataset.text_preprocessor.preprocess_answer(raw_sample['answer'])
    else:
        question_processed = {'cleaned': raw_sample['question'], 'tokens': []}
        answer_processed = {'cleaned': raw_sample['answer'], 'tokens': []}
    
    print("\n=== Text Comparison ===")
    print(f"Raw question: {raw_sample['question']}")
    print(f"Cleaned question: {question_processed['cleaned']}")
    print(f"Question tokens ({len(question_processed.get('tokens', []))}): {question_processed.get('tokens', [])[:20]}")
    
    print(f"\nRaw answer: {raw_sample['answer']}")
    print(f"Cleaned answer: {answer_processed['cleaned']}")
    print(f"Answer tokens ({len(answer_processed.get('tokens', []))}): {answer_processed.get('tokens', [])[:20]}")
    
    print(f"\nProcessed question tensor shape: {processed_sample['question']['input_ids'].shape}")
    if 'input_ids' in processed_sample['answer']:
        print(f"Processed answer tensor shape: {processed_sample['answer']['input_ids'].shape}")

def test_combined_preprocessing():
    """Test the combined preprocessing functionality."""
    print("=== Testing Combined Preprocessing ===")
    
    # Load dataset ONCE
    from datasets import load_dataset
    base_dataset = load_dataset("flaviagiammarino/vqa-rad")['train']

    # Create combined data loaders
    train_loader, val_loader, test_loader, answer_vocab, split_indices = create_combined_data_loaders(
        batch_size=4,
        num_workers=0,  # Use 0 for testing
        max_samples=20,
        base_dataset=base_dataset
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Question text: {batch['question']['text'][0]}")
    print(f"Answer text: {batch['answer']['text'][0]}")
    print(f"Question input_ids shape: {batch['question']['input_ids'].shape}")
    if 'input_ids' in batch['answer']:
        print(f"Answer input_ids shape: {batch['answer']['input_ids'].shape}")
    print(f"Question type sample: {batch.get('question_type', ['unknown'])[0]}")
    print(f"Answer idx sample: {batch['answer_idx'][0].item()}")
    
    # Visualize batch
    visualize_combined_batch(batch)
    
    print(f"Coverage curve points: {len(train_loader.dataset.coverage_curve)} (first 3 shown)")
    for point in train_loader.dataset.coverage_curve[:3]:
        print(f"  k={point['k']}, coverage={point['coverage']:.3f}, answer='{point['answer']}'")
    
    compare_preprocessing_sample(train_loader.dataset, sample_idx=0)
    compare_combined_sample(train_loader.dataset, sample_idx=0)
    
    return train_loader, val_loader, test_loader, answer_vocab, split_indices

def analyze_combined_data():
    """Analyze the combined dataset."""
    print("\n=== Analyzing Combined Data ===")
    
    # Load dataset
    ds = load_dataset("flaviagiammarino/vqa-rad")
    
    # Get sample data
    questions = [ds['train'][i]['question'] for i in range(100)]
    answers = [ds['train'][i]['answer'] for i in range(100)]
    
    # Initialize preprocessors
    image_preprocessor = MedicalImagePreprocessor()
    text_preprocessor = MedicalTextPreprocessor()
    
    # Analyze text data
    print("Text analysis:")
    q_stats = text_preprocessor.analyze_text_statistics(questions)
    a_stats = text_preprocessor.analyze_text_statistics(answers)
    
    print(f"  Question stats: avg={q_stats['avg_length']:.2f}, std={q_stats['std_length']:.2f}")
    print(f"  Answer stats: avg={a_stats['avg_length']:.2f}, std={a_stats['std_length']:.2f}")
    
    # Analyze medical terms
    all_medical_terms = {'anatomy': [], 'conditions': [], 'procedures': [], 'descriptors': []}
    
    for text in questions + answers:
        medical_terms = text_preprocessor.extract_medical_terms(text)
        for category, terms in medical_terms.items():
            all_medical_terms[category].extend(terms)
    
    print("\nMedical terms found:")
    for category, terms in all_medical_terms.items():
        term_counts = Counter(terms)
        print(f"  {category}: {len(term_counts)} unique terms")
        if term_counts:
            print(f"    Top 3: {list(term_counts.most_common(3))}")
    
    return q_stats, a_stats, all_medical_terms

if __name__ == "__main__":
    # Test the combined preprocessing
    train_loader, val_loader, test_loader, answer_vocab, split_indices = test_combined_preprocessing()
    
    # Analyze the data
    analyze_combined_data()
    
    print("Combined preprocessing test completed!")
