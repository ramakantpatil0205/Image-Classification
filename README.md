# Fruits & Vegetables Image Classification Project

**What this archive contains**
- `dataset_script.py` : Script to automatically download and prepare a balanced dataset (10 fruits + 10 vegetables, 80 images each) from the Fruits-360 Kaggle dataset. **You must configure Kaggle API credentials locally** (instructions below).
- `train_model.py` : Training script using TensorFlow / Keras. Loads the dataset folder, applies preprocessing & augmentation, trains a CNN, saves the model and training plots.
- `requirements.txt` : Python dependencies for the project.
- `README.md` : This file with setup & usage instructions.
- `.gitignore` : ignores large dataset and model files.
- `LICENSE` : MIT License.

## Quick overview / steps (local machine)

1. Install Python 3.9+ and pip.
2. Create and activate a virtualenv (recommended).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure Kaggle API:
   - Go to https://www.kaggle.com -> Account -> Create API Token. Download `kaggle.json`.
   - Place `kaggle.json` at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<You>\.kaggle\kaggle.json` (Windows).
   - Make sure permissions are secure: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac).
5. Run dataset script to download & prepare images (this will download the Fruits-360 dataset and select the requested classes):
   ```bash
   python dataset_script.py --output_dir dataset --per_class 80
   ```
   This creates `dataset/fruits/<class>/*` and `dataset/vegetables/<class>/*` with balanced images.
6. Train model:
   ```bash
   python train_model.py --data_dir dataset --model_out models/fruits_veg_cnn.h5
   ```
7. Inspect saved model, accuracy/loss plots in `outputs/`.

## Notes
- This archive does **not** include the dataset images (to avoid large file sizes and licensing concerns). The `dataset_script.py` will download images from Kaggle when run locally.
- If you prefer a sample small dataset to test quickly, change `--per_class` to 5 in `dataset_script.py` run command.
- See inline comments inside the scripts for more details.
