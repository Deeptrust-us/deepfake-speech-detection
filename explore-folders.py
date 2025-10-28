#!/usr/bin/env python3
"""
Dataset Explorer Script
Analyzes the dataset folder structure, size, languages, and file counts.
"""

import os
from pathlib import Path
import argparse


def get_folder_size(folder_path):
    """Calculate total size of a folder in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, PermissionError):
        pass
    return total_size


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def count_files(folder_path, extensions=None):
    """Count files in a folder by extension."""
    if extensions is None:
        extensions = ['.wav', '.csv', '.bak']
    
    counts = {ext: 0 for ext in extensions}
    try:
        for filename in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, filename)):
                ext = os.path.splitext(filename)[1].lower()
                if ext in counts:
                    counts[ext] += 1
        # Also check subdirectories recursively
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in counts:
                    counts[ext] += 1
    except (OSError, PermissionError):
        pass
    return counts


def explore_dataset(dataset_path):
    """Explore the dataset structure and print information."""
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return
    
    print("=" * 80)
    print("DATASET EXPLORER")
    print("=" * 80)
    print(f"Dataset path: {dataset_path.absolute()}\n")
    
    # Overall statistics
    total_size = get_folder_size(dataset_path)
    print(f"Total dataset size: {format_size(total_size)}\n")
    
    # Explore main categories (fake/real)
    categories = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            categories.append(item.name)
    
    print(f"Main categories: {', '.join(categories)}\n")
    
    # Detailed exploration by category
    for category in sorted(categories):
        category_path = dataset_path / category
        category_size = get_folder_size(category_path)
        
        print("=" * 80)
        print(f"CATEGORY: {category.upper()}")
        print("=" * 80)
        print(f"Category size: {format_size(category_size)}")
        print(f"Category path: {category_path}\n")
        
        # Explore languages in this category
        languages = []
        for item in sorted(category_path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                languages.append(item.name)
        
        print(f"Languages: {', '.join(languages)} ({len(languages)} languages)\n")
        
        # Language statistics
        language_stats = []
        for lang in sorted(languages):
            lang_path = category_path / lang
            lang_size = get_folder_size(lang_path)
            
            # Get file counts
            file_counts = count_files(lang_path)
            total_files = sum(file_counts.values())
            
            # Get subfolder count (TTS models)
            try:
                subfolders = [d for d in lang_path.iterdir() if d.is_dir()]
                tts_models_count = len(subfolders)
                tts_models = sorted([d.name for d in subfolders])[:5]  # First 5 as sample
            except (OSError, PermissionError):
                tts_models_count = 0
                tts_models = []
            
            language_stats.append({
                'name': lang,
                'size': lang_size,
                'total_files': total_files,
                'wav_files': file_counts.get('.wav', 0),
                'csv_files': file_counts.get('.csv', 0),
                'tts_models_count': tts_models_count,
                'tts_models_sample': tts_models
            })
        
        # Print language statistics table
        print("-" * 80)
        print(f"{'Language':<15} {'Size':<15} {'Total Files':<15} {'WAV Files':<15} {'CSV Files':<12} {'TTS Models':<10}")
        print("-" * 80)
        
        for stats in language_stats:
            size_str = format_size(stats['size'])
            print(f"{stats['name']:<15} {size_str:<15} {stats['total_files']:<15} "
                  f"{stats['wav_files']:<15} {stats['csv_files']:<12} {stats['tts_models_count']:<10}")
        
        print("-" * 80)
        print()
        
        # Show detailed structure for first language
        if language_stats:
            first_lang = language_stats[0]
            print(f"Sample: TTS models in '{first_lang['name']}' language:")
            if first_lang['tts_models_sample']:
                print("  " + "\n  ".join(first_lang['tts_models_sample']))
                if first_lang['tts_models_count'] > len(first_lang['tts_models_sample']):
                    print(f"  ... and {first_lang['tts_models_count'] - len(first_lang['tts_models_sample'])} more")
            print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_languages = set()
    total_wav_files = 0
    total_csv_files = 0
    total_tts_models = 0
    
    for category in sorted(categories):
        category_path = dataset_path / category
        for lang in sorted(category_path.iterdir()):
            if lang.is_dir() and not lang.name.startswith('.'):
                all_languages.add(lang.name)
                
                file_counts = count_files(lang)
                total_wav_files += file_counts.get('.wav', 0)
                total_csv_files += file_counts.get('.csv', 0)
                
                try:
                    subfolders = [d for d in lang.iterdir() if d.is_dir()]
                    total_tts_models += len(subfolders)
                except (OSError, PermissionError):
                    pass
    
    print(f"Total categories: {len(categories)}")
    print(f"Total languages: {len(all_languages)}")
    print(f"Languages: {', '.join(sorted(all_languages))}")
    print(f"Total WAV files: {total_wav_files:,}")
    print(f"Total CSV files: {total_csv_files:,}")
    print(f"Total TTS model folders: {total_tts_models:,}")
    print(f"Total dataset size: {format_size(total_size)}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Explore dataset structure, size, and statistics'
    )
    parser.add_argument(
        'dataset_path',
        nargs='?',
        default='dataset',
        help='Path to the dataset folder (default: dataset)'
    )
    
    args = parser.parse_args()
    explore_dataset(args.dataset_path)


if __name__ == '__main__':
    main()

