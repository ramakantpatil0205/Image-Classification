"""dataset_script.py
Downloads Fruits-360 dataset from Kaggle, selects 10 fruits and 10 vegetables
and prepares a balanced dataset with N images per class (default 80).
Usage:
    python dataset_script.py --output_dir dataset --per_class 80
"""
import argparse
import os
import shutil
from pathlib import Path
from collections import defaultdict
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import random

# Default class lists (you can edit these)
FRUITS = [
    'Apple', 'Apricot', 'Avocado', 'Banana', 'Cherry', 'Grape', 'Lemon',
    'Mango', 'Orange', 'Pineapple'
]
VEGETABLES = [
    'Beetroot', 'Broccoli', 'Cabbage', 'Carrot', 'Cauliflower', 'Cucumber',
    'Eggplant', 'Onion', 'Pepper', 'Tomato'
]

def safe_name(n):
    # Fruits-360 uses names like 'Apple Braeburn' etc. This helper does case-insensitive matching.
    return n.strip().lower()

def collect_classes_from_zip(zip_path):
    # List top-level folders inside the zip's 'Training' folder
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = z.namelist()
    classes = set()
    prefix = 'Training/'
    for p in names:
        if p.startswith(prefix) and len(p) > len(prefix):
            parts = p[len(prefix):].split('/')
            if len(parts) >= 1 and parts[0]:
                classes.add(parts[0])
    return classes

def main(output_dir='dataset', per_class=80, kaggle_dataset='moltean/fruits'):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print('Downloading Kaggle dataset:', kaggle_dataset)
    # downloads to current working directory a zip file or folder
    api.dataset_download_files(kaggle_dataset, path='.', unzip=False, quiet=False)

    zip_name = 'fruits.zip'
    # Kaggle typically names the zip after dataset; find the first .zip in cwd
    zips = [p for p in Path('.').iterdir() if p.suffix == '.zip']
    if not zips:
        raise FileNotFoundError('No zip found after Kaggle download. Check dataset id and kaggle.json.')
    zip_path = str(sorted(zips, key=lambda p: p.stat().st_mtime)[-1])
    print('Found zip:', zip_path)

    classes_in_zip = collect_classes_from_zip(zip_path)
    print('Classes found in zip (sample):', list(classes_in_zip)[:20])

    # map desired class names to matching names in dataset (case-insensitive)
    matched_fruits = []
    matched_vegs = []
    lower_classes = {c.lower(): c for c in classes_in_zip}
    for target in FRUITS:
        # find closest match by exact lower-case or substring
        t = target.lower()
        candidates = [orig for orig in classes_in_zip if t in orig.lower()]
        if candidates:
            matched_fruits.append(candidates[0])
    for target in VEGETABLES:
        t = target.lower()
        candidates = [orig for orig in classes_in_zip if t in orig.lower()]
        if candidates:
            matched_vegs.append(candidates[0])

    selected = matched_fruits + matched_vegs
    if len(selected) < 20:
        print('Warning: matched only', len(selected), 'classes. You may need to edit FRUITS/VEGETABLES lists.')

    # Extract only required files into the output structure
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = [m for m in z.namelist() if any(('/Training/'+c+'/') in m for c in selected)]
        print('Total matching files in training folders:', len(members))
        # create class folders
        for c in selected:
            # determine category
            cat = 'fruits' if c in matched_fruits else 'vegetables'
            target_dir = out / cat / c
            target_dir.mkdir(parents=True, exist_ok=True)

        # group files by class
        grouped = defaultdict(list)
        for m in members:
            parts = m.split('/')
            if len(parts) >= 3 and parts[0] == 'Training':
                cls = parts[1]
                grouped[cls].append(m)

        for cls, files_list in grouped.items():
            random.shuffle(files_list)
            take = files_list[:per_class]
            for f in take:
                dest = out / ('fruits' if cls in matched_fruits else 'vegetables') / cls / Path(f).name
                with z.open(f) as src, open(dest, 'wb') as dst:
                    dst.write(src.read())

    print('Dataset prepared at', out.resolve())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='dataset')
    parser.add_argument('--per_class', type=int, default=80)
    args = parser.parse_args()
    main(output_dir=args.output_dir, per_class=args.per_class)
