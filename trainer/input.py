# trainer/input.py
import tensorflow as tf
import os
import functools # <--- Add this import for partial function application
import sys # Import sys for installation check

# Use the TensorFlow wrapped version
# Assuming tf_rgb_to_lab_normalized is correctly defined in utils
from .utils import tf_rgb_to_lab_normalized

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# --- FINAL parse_image_pair (Outputting LAB) ---
def parse_image_pair(low_path, high_path, is_training=False):
    """Loads image, processes RGB, converts to LAB, returns LAB."""
    try:
        # --- 1. Read image files ---
        low_img_str = tf.io.read_file(low_path)
        high_img_str = tf.io.read_file(high_path)

        # --- 2. Decode images ---
        low_img_decoded = tf.image.decode_png(low_img_str, channels=IMG_CHANNELS, dtype=tf.uint8)
        high_img_decoded = tf.image.decode_png(high_img_str, channels=IMG_CHANNELS, dtype=tf.uint8)
        low_img_decoded = tf.ensure_shape(low_img_decoded, [None, None, IMG_CHANNELS])
        high_img_decoded = tf.ensure_shape(high_img_decoded, [None, None, IMG_CHANNELS])

        # --- 3. Convert to float32 and Scale to [0, 1] ---
        low_img_float = tf.cast(low_img_decoded, tf.float32) / 255.0
        high_img_float = tf.cast(high_img_decoded, tf.float32) / 255.0
        low_img_float = tf.clip_by_value(low_img_float, 0.0, 1.0)
        high_img_float = tf.clip_by_value(high_img_float, 0.0, 1.0)

        # --- 4. Resize images ---
        low_img_resized = tf.image.resize(low_img_float, [IMG_HEIGHT, IMG_WIDTH])
        high_img_resized = tf.image.resize(high_img_float, [IMG_HEIGHT, IMG_WIDTH])

        # --- 5. Apply augmentation (if training) ---
        low_img_final_rgb = low_img_resized # Assign to final RGB variable
        high_img_final_rgb = high_img_resized

        if is_training:
            low_img_final_rgb = tf.image.random_brightness(low_img_final_rgb, max_delta=0.1)
            low_img_final_rgb = tf.image.random_contrast(low_img_final_rgb, lower=0.9, upper=1.1)
            low_img_final_rgb = tf.clip_by_value(low_img_final_rgb, 0.0, 1.0)

        # --- 6. Convert FINAL RGB to Normalized LAB ---
        # Expand dimensions for the wrapper function
        low_img_expanded = tf.expand_dims(low_img_final_rgb, 0)
        high_img_expanded = tf.expand_dims(high_img_final_rgb, 0)

        # Use the TF wrapper function from utils.py
        low_lab = tf_rgb_to_lab_normalized(low_img_expanded)
        high_lab = tf_rgb_to_lab_normalized(high_img_expanded)

        # Remove the batch dimension
        low_lab = tf.squeeze(low_lab, axis=0)
        high_lab = tf.squeeze(high_lab, axis=0)

        # Ensure shape consistency before returning
        low_lab = tf.ensure_shape(low_lab, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
        high_lab = tf.ensure_shape(high_lab, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])

        return low_lab, high_lab # Return LAB tensors

    except Exception as e:
        tf.print(f"Error processing pair: {low_path}, {high_path}. Error: {e}", output_stream=sys.stderr)
        dummy_lab = tf.zeros([IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=tf.float32)
        return dummy_lab, dummy_lab

def load_dataset(low_dir, high_dir, batch_size, is_training=True):
    """
    Loads a dataset (train or validation) using tf.data.
    Args:
        low_dir (str): Path to low-quality images directory.
        high_dir (str): Path to high-quality target images directory.
        batch_size (int): Batch size.
        is_training (bool): If True, shuffle and augment the dataset.
    Returns:
        tf.data.Dataset: A batched dataset of (low_lab, high_lab) image pairs.
    """
    try:
        low_dir = low_dir.rstrip('/')
        high_dir = high_dir.rstrip('/')

        low_files_pattern = os.path.join(low_dir, '*.png')
        high_files_pattern = os.path.join(high_dir, '*.png')

        print(f"Looking for low-quality images at: {low_files_pattern}")
        print(f"Looking for high-quality images at: {high_files_pattern}")

        low_files = tf.data.Dataset.list_files(low_files_pattern, shuffle=False)
        high_files = tf.data.Dataset.list_files(high_files_pattern, shuffle=False)

        # Check if any files were found
        if tf.data.experimental.cardinality(low_files) == 0:
             tf.print(f"Warning: No files found matching pattern {low_files_pattern}", output_stream=tf.compat.v1.logging.WARN)
             return tf.data.Dataset.from_tensor_slices((
                 tf.zeros([0, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=tf.float32),
                 tf.zeros([0, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=tf.float32)
             )).batch(batch_size)

        # Simple check for file count matching
        tf.debugging.assert_equal(tf.data.experimental.cardinality(low_files),
                                  tf.data.experimental.cardinality(high_files),
                                  message=f"Number of low ({low_dir}) and high ({high_dir}) quality images must match.")

        dataset = tf.data.Dataset.zip((low_files, high_files))

        if is_training:
            dataset_size = tf.data.experimental.cardinality(dataset)
            buffer_size = dataset_size if dataset_size != tf.data.UNKNOWN_CARDINALITY and dataset_size > 0 else tf.constant(1000, dtype=tf.int64) # Handle unknown size
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        # Use AUTOTUNE for optimal parallel calls and prefetching
        # ---> MODIFICATION 3: Pass is_training to parse_image_pair using partial ---
        # Create a partial function with the is_training argument fixed
        parse_fn = functools.partial(parse_image_pair, is_training=is_training)
        # Apply the partial function using map
        dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        # --- End Modification ---

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        print(f"[*] Dataset loaded from {low_dir} and {high_dir}. Training mode: {is_training}")
        print(f"[*] Batch size: {batch_size}")

        return dataset

    except Exception as e:
        print(f"Error loading dataset from {low_dir}/{high_dir}: {e}")
        # Return an empty dataset in case of broader loading errors
        return tf.data.Dataset.from_tensor_slices((
             tf.zeros([0, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=tf.float32),
             tf.zeros([0, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS], dtype=tf.float32)
         )).batch(batch_size)