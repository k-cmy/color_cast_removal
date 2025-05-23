# Color Cast Removal using Deep Learning

This project implements a deep learning model using TensorFlow and Keras to automatically remove color casts from images. The model learns to transform images with unnatural color tints (e.g., too yellow, too blue) into images with more natural-looking colors, using pairs of low-quality (casted) and high-quality (ground truth) images for training.

## Project Structure

```
color_cast_removal_trainer-2/
├── data/
│   ├── train/             # Training data (or use root data/ folder)
│   │   ├── low/           # Low-quality images with color cast
│   │   └── high/          # Corresponding high-quality ground truth images
│   └── test/              # Test data (used for evaluation)
│       ├── low/
│       └── high/
├── trainer/               # Main package for the training code
│   ├── __init__.py       # Python recognizes it as a package
│   ├── input.py         # Data loading and preprocessing pipeline
│   ├── model.py         # Defines the ColorCastRemoval Keras model
│   ├── task.py     # Main training and validation logic (for Google Cloud Vertex AI)
│   ├── taskLOCAL.py    # Main training and validation logic (for local env testing)
│   └── utils.py         # Utility functions (e.g., color space conversions)
├── local_output/          # Default directory for saving checkpoints, logs, and final model
│   ├── checkpoints/
│   ├── logs/
│   └── final_model/
├── evaluation_output_images/ # Default directory for saving evaluation results
├── local_test.py        # Script to run training locally
├── evaluate_model.py    # Script to evaluate a trained model
├── inference.py         # Script for inference on new images (NEW)
├── preprocess_images.py # Optional script to fix PNG profiles
├── README.md         # This file
├── requirements.txt    # List of required packages
└── Dockerfile         # Dockerfile for containerized training (optional)
```

## Setup

1. **Clone the Repository:** (If applicable)
   ```bash
   git clone https://github.com/k-cmy/color_cast_removal.git
   cd color_cast_removal_trainer-2
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.x installed. Install the required libraries:
   ```bash
   pip install tensorflow numpy scikit-image matplotlib opencv-python Pillow
   ```
   *(Optional: Consider creating a `requirements.txt` file for easier environment setup: `pip freeze > requirements.txt` and then `pip install -r requirements.txt`)*

3. **Prepare Data:**
   * Create the necessary data directories:
     * `data/train/low/`
     * `data/train/high/`
     * `data/test/low/`
     * `data/test/high/`
     
     *(Adjust paths in `local_test.py` and `evaluate_model.py` if you use a different structure).*
   
   * Populate these directories with your paired **PNG** images. Ensure filenames correspond between the `low` and `high` directories for each pair.
   
   * **(Optional) Preprocess PNGs:** If you suspect issues with image loading due to ICC profiles, run:
     ```bash
     python preprocess_images.py
     ```
     **⚠️ Warning:** This script modifies images in place. **Backup your original data before running!**

## Training

The `local_test.py` script handles the training process using default settings and data paths.

**To start training:**

```bash
python local_test.py
```

### Notes:
- **Configuration:** Modify hyperparameters (epochs, batch size, learning rate, etc.) directly within the `local_test.py` script if needed.
- **Checkpoints:** Model checkpoints are saved periodically to `local_output/checkpoints/`. Training can be resumed automatically if checkpoints exist.
- **Logs:** View training progress using TensorBoard:
  ```bash
  tensorboard --logdir local_output/logs/
  ```
- **Final Model:** The fully trained model is saved as `local_output/final_model/model.keras`.

## Inference

After training, you can use the trained model to perform inference on new images.

**Example Command:**

Windows (cmd/powershell):
```bash
python inference_pipeline.py `
--model-path local_output/final_model/model2.keras ` 
--input-path data/test/low --output-dir inference_results `
--ground-truth-dir data/test/high
```
Linux/macOS:
```bash
python inference_pipeline.py \
--model-path local_output/final_model/model2.keras \
--input-path data/test/low --output-dir inference_results \
--ground-truth-dir data/test/high
```
### Arguments:
- `--model-path`: Path to the trained model file.
- `--input-path`: Path to the low-quality images for inference.
- `--output-dir`: Directory to save the inference results.
- `--ground-truth-dir`: Path to the high-quality ground truth images (optional, for comparison).

## Evaluation

After training, evaluate the model's performance on the test dataset.

**Example Command:**

Windows (cmd/powershell):
```bash
python evaluate_model.py `
  --model-path local_output/final_model/model.keras `
  --test-low-dir data/test/low `
  --test-high-dir data/test/high `
  --batch-size 8 `
  --output-dir evaluation_output_images
```

Linux/macOS:
```bash
python evaluate_model.py \
  --model-path local_output/final_model/model.keras \
  --test-low-dir data/test/low \
  --test-high-dir data/test/high \
  --batch-size 8 \
  --output-dir evaluation_output_images
```

### Arguments:
- `--model-path`: Path to the saved .keras model file.
- `--test-low-dir`: Path to the low-quality test images.
- `--test-high-dir`: Path to the high-quality ground truth test images.
- `--batch-size`: Evaluation batch size (adjust based on memory).