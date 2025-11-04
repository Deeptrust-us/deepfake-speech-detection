"""
MultilingualDataset - Loads the new organized multilingual deepfake speech detection dataset.

This dataset loads from labels.json which contains metadata about audio files
organized in the dataset_audios/audio/real/ and dataset_audios/audio/fake/ structure.
"""

import os
import json
from pathlib import Path
from typing import List, Optional

from ._dataclass import DF_Item


class MultilingualDataset:
    """
    Dataset loader for the multilingual deepfake speech detection dataset.
    
    This dataset loads audio files from the organized structure:
    - dataset_audios/audio/real/ contains real audio files
    - dataset_audios/audio/fake/ contains fake audio files
    - labels.json contains metadata for all audio files
    """
    
    def __init__(
        self,
        labels_path: str,
        dataset_root: Optional[str] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
        print_info: bool = False
    ):
        """
        Initialize the MultilingualDataset.
        
        Args:
            labels_path: Path to labels.json file
            dataset_root: Root directory of the dataset (default: parent of labels.json)
            train_split: Fraction of data for training (default: 0.8)
            val_split: Fraction of data for validation (default: 0.1)
            test_split: Fraction of data for testing (default: 0.1)
            random_seed: Random seed for splitting (default: 42)
            print_info: Whether to print dataset information (default: False)
        """
        import random
        import numpy as np
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load labels.json
        labels_path = Path(labels_path).resolve()
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        
        # Determine dataset root
        if dataset_root is None:
            dataset_root = labels_path.parent
        else:
            dataset_root = Path(dataset_root).resolve()
        
        self.dataset_root = dataset_root
        
        # Initialize sets
        self.train_set = []
        self.val_set = []
        self.test_set = []
        self.class_weight = []
        
        # Convert labels to DF_Item objects
        items = []
        train_num_pos = 0  # fake (label=1)
        train_num_neg = 0  # real (label=0)
        
        for entry in self.labels:
            # Get audio file path
            # Priority: check if file exists in audio/real or audio/fake
            # Otherwise, use original_path
            label_str = entry.get('label', 'real')
            filename = entry.get('filename', '')
            
            # Try to find file in organized structure first
            if label_str == 'real':
                audio_path = self.dataset_root / 'audio' / 'real' / filename
            else:
                audio_path = self.dataset_root / 'audio' / 'fake' / filename
            
            # Fallback to original_path if organized file doesn't exist
            if not audio_path.exists():
                original_path = entry.get('original_path', '')
                if original_path:
                    audio_path = Path(original_path)
                else:
                    # Skip if no valid path found
                    continue
            
            # Convert label: 'real' -> 0, 'fake' -> 1
            label = 0 if label_str == 'real' else 1
            
            # Get model/speaker as attack_type
            attack_type = entry.get('model_or_speaker', 'unknown')
            if label == 0:
                attack_type = 'real'  # For real samples, use 'real' as attack_type
            
            # Count for class weights
            if label == 0:
                train_num_neg += 1
            else:
                train_num_pos += 1
            
            # Create DF_Item
            item = DF_Item(
                path=str(audio_path.resolve()),
                label=label,
                attack_type=attack_type,
                is_fake=(label == 1)
            )
            items.append(item)
        
        # Shuffle items for train/val/test split
        random.shuffle(items)
        
        # Split dataset
        total = len(items)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        
        self.train_set = items[:train_end]
        self.val_set = items[train_end:val_end]
        self.test_set = items[val_end:]
        
        # Calculate class weights for balancing
        if train_num_neg > 0 and train_num_pos > 0:
            total_count = train_num_neg + train_num_pos
            self.class_weight.append(total_count / train_num_neg)  # weight for real (label=0)
            self.class_weight.append(total_count / train_num_pos)  # weight for fake (label=1)
        else:
            self.class_weight = [1.0, 1.0]
        
        # Print info if requested
        if print_info:
            train_real = sum(1 for item in self.train_set if item.label == 0)
            train_fake = sum(1 for item in self.train_set if item.label == 1)
            val_real = sum(1 for item in self.val_set if item.label == 0)
            val_fake = sum(1 for item in self.val_set if item.label == 1)
            test_real = sum(1 for item in self.test_set if item.label == 0)
            test_fake = sum(1 for item in self.test_set if item.label == 1)
            
            info = (
                f'====================\n'
                + f'  MultilingualDataset\n'
                + f'====================\n'
                + f'TRAIN: Real={train_real:,}, Fake={train_fake:,}, Total={len(self.train_set):,}\n'
                + f'VAL:   Real={val_real:,}, Fake={val_fake:,}, Total={len(self.val_set):,}\n'
                + f'TEST:  Real={test_real:,}, Fake={test_fake:,}, Total={len(self.test_set):,}\n'
                + f'Class weights: Real={self.class_weight[0]:.4f}, Fake={self.class_weight[1]:.4f}\n'
                + f'====================\n'
            )
            print(info)

