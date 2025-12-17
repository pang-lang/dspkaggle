#!/usr/bin/env python3

import json
import os
from typing import Tuple, Optional, Union, List, Dict

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEFAULT_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

DEFAULT_STATS_DIR = os.path.join('data_splits', 'stats')
DEFAULT_STATS_SEED = 42
DEFAULT_STATS_CACHE_PATH = os.path.join(DEFAULT_STATS_DIR, f'stats_seed{DEFAULT_STATS_SEED}.json')


class MedicalImagePreprocessor:
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 use_albumentations: bool = False,
                 normalization_stats: Optional[Dict[str, List[float]]] = None):
        self.target_size = target_size
        self.use_albumentations = use_albumentations
        if normalization_stats is None:
            self.normalization_stats = {
                'mean': list(DEFAULT_NORMALIZATION['mean']),
                'std': list(DEFAULT_NORMALIZATION['std'])
            }
        else:
            self.normalization_stats = {
                'mean': list(normalization_stats.get('mean', DEFAULT_NORMALIZATION['mean'])),
                'std': list(normalization_stats.get('std', DEFAULT_NORMALIZATION['std']))
            }
        self._torch_transform_train = None
        self._torch_transform_eval = None
        self._alb_transform_train = None
        self._alb_transform_eval = None

    def set_normalization_stats(self, mean: List[float], std: List[float]):
        """Update normalization stats (e.g., dataset-level) and reset cached transforms."""
        self.normalization_stats = {'mean': list(mean), 'std': list(std)}
        self._reset_cached_transforms()

    def _reset_cached_transforms(self):
        self._torch_transform_train = None
        self._torch_transform_eval = None
        self._alb_transform_train = None
        self._alb_transform_eval = None

    def _get_normalization_params(self) -> Tuple[List[float], List[float]]:
        mean = self.normalization_stats.get('mean', DEFAULT_NORMALIZATION['mean'])
        std = self.normalization_stats.get('std', DEFAULT_NORMALIZATION['std'])
        return mean, std
        
    def resize_image(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        if size is None:
            size = self.target_size
        
        # Calculate aspect ratio preserving resize
        w, h = image.size
        target_w, target_h = size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        new_image = Image.new('RGB', size, (0, 0, 0))
        new_image.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
        
        return new_image
    
    def enhance_contrast(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def enhance_brightness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def enhance_sharpness(self, image: Image.Image, factor: float = 1.5) -> Image.Image:
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            return clahe.apply(image)
    
    def denoise_image(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        if method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
    
    def apply_gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(image, table)
    
    def extract_roi(self, image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create binary mask
        _, mask = cv2.threshold(gray, threshold * 255, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract ROI
            return image[y:y+h, x:x+w]
        
        return image
    
    def create_pytorch_transforms(self, is_training: bool = True) -> transforms.Compose:
        """Create PyTorch transforms for training/inference."""
        mean, std = self._get_normalization_params()
        if is_training:
            transform_list = [
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=3),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    shear=5
                ),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        else:
            transform_list = [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        
        return transforms.Compose(transform_list)
    
    def create_albumentations_transform(self, is_training: bool = True) -> A.Compose:
        """Create Albumentations transforms for advanced augmentation."""
        mean, std = self._get_normalization_params()
        if is_training:
            transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.HorizontalFlip(p=0.4),
                A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.05),
                         rotate=(-3, 3), shear=(-5, 5), mode=cv2.BORDER_CONSTANT, p=0.4),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
                A.MotionBlur(blur_limit=3, p=0.1),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(self.target_size[0], self.target_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        
        return transform
    
    def _get_torch_transform(self, is_training: bool):
        attr = '_torch_transform_train' if is_training else '_torch_transform_eval'
        transform = getattr(self, attr)
        if transform is None:
            transform = self.create_pytorch_transforms(is_training)
            setattr(self, attr, transform)
        return transform
    
    def _get_alb_transform(self, is_training: bool):
        attr = '_alb_transform_train' if is_training else '_alb_transform_eval'
        transform = getattr(self, attr)
        if transform is None:
            transform = self.create_albumentations_transform(is_training)
            setattr(self, attr, transform)
        return transform
    
    def prepare_image(self,
                      image: Union[Image.Image, np.ndarray],
                      is_training: bool = True,
                      preprocess_steps: Optional[List[str]] = None) -> torch.Tensor:
        """
        Convert a raw PIL/NumPy image into a normalized tensor using optional preprocessing.
        This avoids running both classical steps and torchvision transforms separately.
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        if preprocess_steps:
            image_np = self.preprocess_pipeline(image_np, preprocess_steps)
        
        letterboxed_pil = self.resize_image(Image.fromarray(image_np.astype(np.uint8)))
        letterboxed_np = np.array(letterboxed_pil)
        
        if self.use_albumentations:
            transform = self._get_alb_transform(is_training)
            augmented = transform(image=letterboxed_np)
            tensor = augmented['image']
        else:
            transform = self._get_torch_transform(is_training)
            tensor = transform(letterboxed_pil)
        
        return tensor
    
    def preprocess_pipeline(self, image: Union[Image.Image, np.ndarray], 
                          steps: List[str] = None) -> np.ndarray:
        """Apply a complete preprocessing pipeline. Default is no-op to avoid double normalization."""
        if steps is None:
            steps = []
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        processed_image = image.copy()
        
        for step in steps:
            if step == 'resize':
                # Convert back to PIL for resize, then back to numpy
                pil_image = Image.fromarray(processed_image)
                resized = self.resize_image(pil_image)
                processed_image = np.array(resized)
            
            elif step == 'enhance_contrast':
                pil_image = Image.fromarray(processed_image)
                enhanced = self.enhance_contrast(pil_image)
                processed_image = np.array(enhanced)
            
            elif step == 'denoise':
                processed_image = self.denoise_image(processed_image)
            
            elif step == 'clahe':
                processed_image = self.apply_clahe(processed_image)
            
            elif step == 'gamma':
                processed_image = self.apply_gamma_correction(processed_image, gamma=1.2)
            
            elif step == 'roi':
                processed_image = self.extract_roi(processed_image)
            
            elif step == 'normalize':
                # Per-image min-max normalization to [0, 255]
                img_float = processed_image.astype(np.float32)
                img_min = img_float.min()
                img_max = img_float.max()
                if img_max > img_min:
                    img_float = (img_float - img_min) / (img_max - img_min)
                    img_float *= 255.0
                else:
                    img_float = np.zeros_like(img_float)
                processed_image = img_float.astype(np.uint8)
        
        return processed_image
    
    def batch_preprocess_images(self, images: List[Union[Image.Image, np.ndarray]], 
                              steps: List[str] = None) -> List[np.ndarray]:
        """Preprocess a batch of images."""
        processed_images = []
        
        for image in images:
            processed = self.preprocess_pipeline(image, steps)
            processed_images.append(processed)
        
        return processed_images


def _sanitize_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace('/', '_').replace('-', '_')


def _default_stats_cache_path(split_seed: int = DEFAULT_STATS_SEED) -> str:
    os.makedirs(DEFAULT_STATS_DIR, exist_ok=True)
    return os.path.join(DEFAULT_STATS_DIR, f'stats_seed{split_seed}.json')


def load_cached_dataset_stats(cache_path: Optional[str],
                              expected_meta: Optional[Dict] = None) -> Optional[Dict[str, List[float]]]:
    if not cache_path or not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r') as fp:
        stats = json.load(fp)
    if 'mean' not in stats or 'std' not in stats:
        return None
    if expected_meta:
        meta = stats.get('meta', {})
        mismatch = any(meta.get(k) != v for k, v in expected_meta.items())
        if mismatch:
            return None
    return stats


def compute_dataset_statistics(dataset_name: str = "flaviagiammarino/vqa-rad",
                               split: str = "train",
                               target_size: Tuple[int, int] = (224, 224),
                               preprocess_steps: Optional[List[str]] = None,
                               max_samples: Optional[int] = None,
                               cache_path: Optional[str] = None,
                               split_seed: int = DEFAULT_STATS_SEED) -> Dict[str, Union[List[float], int]]:
    """
    Compute dataset-level mean and std after resizing (before normalization).
    """
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name)[split]
    total_images = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    preprocessor = MedicalImagePreprocessor(target_size=target_size, use_albumentations=False)
    channel_sum = torch.zeros(3)
    channel_sq_sum = torch.zeros(3)
    total_pixels = 0
    processed = 0
    
    for idx in tqdm(range(total_images), desc=f"Computing stats ({split})"):
        sample = dataset[idx]
        image = np.array(sample['image'])
        if preprocess_steps:
            image = preprocessor.preprocess_pipeline(image, preprocess_steps)
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized = preprocessor.resize_image(pil_image)
        tensor = F.to_tensor(resized)
        c, h, w = tensor.shape
        tensor = tensor.view(c, -1)
        channel_sum += tensor.sum(dim=1)
        channel_sq_sum += (tensor ** 2).sum(dim=1)
        total_pixels += h * w
        processed += 1
    
    if processed == 0 or total_pixels == 0:
        return {
            'mean': list(DEFAULT_NORMALIZATION['mean']),
            'std': list(DEFAULT_NORMALIZATION['std']),
            'num_images': 0,
            'split': split
        }
    
    mean = channel_sum / total_pixels
    variance = (channel_sq_sum / total_pixels) - mean ** 2
    variance = torch.clamp(variance, min=1e-12)
    std = torch.sqrt(variance)
    stats = {
        'mean': [float(m) for m in mean],
        'std': [float(s) for s in std],
        'num_images': processed,
        'split': split,
        'meta': {
            'dataset_name': dataset_name,
            'target_size': list(target_size),
            'preprocess_steps': preprocess_steps or [],
            'split_seed': split_seed
        }
    }
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as fp:
            json.dump(stats, fp, indent=2)
    
    return stats


def get_or_compute_dataset_stats(dataset_name: str = "flaviagiammarino/vqa-rad",
                                 split: str = "train",
                                 target_size: Tuple[int, int] = (224, 224),
                                 preprocess_steps: Optional[List[str]] = None,
                                 max_samples: Optional[int] = None,
                                 cache_path: Optional[str] = None,
                                 force_recompute: bool = False,
                                 split_seed: int = DEFAULT_STATS_SEED) -> Dict[str, Union[List[float], int]]:
    """
    Convenience helper to load cached stats or compute them if missing.
    """
    if cache_path is None:
        cache_path = _default_stats_cache_path(split_seed)
    expected_meta = {
        'dataset_name': dataset_name,
        'target_size': list(target_size),
        'preprocess_steps': preprocess_steps or [],
        'split_seed': split_seed
    }
    cached = None if force_recompute else load_cached_dataset_stats(cache_path, expected_meta)
    if cached:
        return cached
    return compute_dataset_statistics(
        dataset_name=dataset_name,
        split=split,
        target_size=target_size,
        preprocess_steps=preprocess_steps,
        max_samples=max_samples,
        cache_path=cache_path,
        split_seed=split_seed
    )

def visualize_preprocessing_results(original_image: np.ndarray, 
                                  processed_images: dict, 
                                  titles: dict = None):
    """Visualize original and processed images side by side."""
    n_images = len(processed_images) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4 * n_images, 4))
    
    if n_images == 1:
        axes = [axes]
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Processed images
    for i, (method, processed) in enumerate(processed_images.items(), 1):
        axes[i].imshow(processed)
        title = titles.get(method, method) if titles else method
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage and testing functions
def test_preprocessing():
    """Test the preprocessing functions with sample data."""
    from datasets import load_dataset
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("flaviagiammarino/vqa-rad")
    
    # Get a sample image
    sample = ds['train'][0]
    image = np.array(sample['image'])
    
    print(f"Original image shape: {image.shape}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    
    # Initialize preprocessor
    preprocessor = MedicalImagePreprocessor(target_size=(224, 224))
    
    # Test different preprocessing methods
    processed_images = {}
    
    # Basic preprocessing
    processed_images['resized'] = preprocessor.preprocess_pipeline(image, ['resize'])
    processed_images['normalized'] = preprocessor.preprocess_pipeline(image, ['resize', 'normalize'])
    processed_images['enhanced'] = preprocessor.preprocess_pipeline(image, ['resize', 'enhance_contrast'])
    processed_images['denoised'] = preprocessor.preprocess_pipeline(image, ['resize', 'denoise'])
    processed_images['clahe'] = preprocessor.preprocess_pipeline(image, ['resize', 'clahe'])
    
    # Visualize results
    titles = {
        'resized': 'Resized',
        'normalized': 'Normalized',
        'enhanced': 'Contrast Enhanced',
        'denoised': 'Denoised',
        'clahe': 'CLAHE'
    }
    
    visualize_preprocessing_results(image, processed_images, titles)
    
    return preprocessor, processed_images

if __name__ == "__main__":
    # Test the preprocessing
    preprocessor, results = test_preprocessing()
    print("Preprocessing test completed!")
