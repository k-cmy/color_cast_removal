# trainer/utils.py
import numpy as np
from skimage import color
import tensorflow as tf
import warnings # Import warnings module

# --- Numpy function for RGB to Normalized LAB ---
# This function seems okay, input RGB is clipped before conversion.
def rgb_to_lab_normalized(rgb_image_np):
    """Convert RGB [0,1] numpy array to normalized LAB [0,1] numpy array."""
    was_single_image = False
    if rgb_image_np.ndim == 3:
        rgb_image_np = np.expand_dims(rgb_image_np, axis=0)
        was_single_image = True

    batch_lab = []
    for img_rgb in rgb_image_np:
        img_rgb_clipped = np.clip(img_rgb, 0.0, 1.0).astype(np.float64)
        lab = color.rgb2lab(img_rgb_clipped)

        lab[..., 0] = lab[..., 0] / 100.0
        lab[..., 1:] = (lab[..., 1:] + 128.0) / 255.0
        lab = np.clip(lab, 0.0, 1.0)
        batch_lab.append(lab)

    lab_batch = np.stack(batch_lab, axis=0)
    if was_single_image:
        lab_batch = lab_batch[0]
    return lab_batch.astype(np.float32)


# --- Numpy function for Normalized LAB to RGB (with added NaN handling and clipping) ---
def numpy_lab_normalized_to_rgb_clipped(lab_normalized_np):
    """
    Convert normalized LAB [0,1] numpy array back to RGB [0,1] numpy array.
    Includes clipping steps and NaN handling.
    """
    lab_normalized_np = np.clip(lab_normalized_np, 0.0, 1.0) # Clip input normalized LAB first

    was_single_image = False
    if lab_normalized_np.ndim == 3:
        lab_normalized_np = np.expand_dims(lab_normalized_np, axis=0)
        was_single_image = True

    batch_rgb = []
    for lab_norm in lab_normalized_np:
        lab = lab_norm.astype(np.float64)
        denorm_lab = np.copy(lab)

        denorm_lab[..., 0] = denorm_lab[..., 0] * 100.0
        denorm_lab[..., 1:] = denorm_lab[..., 1:] * 255.0 - 128.0

        # Clip L to its valid range [0, 100] before lab2rgb conversion
        denorm_lab[..., 0] = np.clip(denorm_lab[..., 0], 0.0, 100.0)
        # Optional: Clip a/b if warnings persist strongly, though this might be overly restrictive
        # denorm_lab[..., 1] = np.clip(denorm_lab[..., 1], -128.0, 127.0) # Theoretical range
        # denorm_lab[..., 2] = np.clip(denorm_lab[..., 2], -128.0, 127.0) # Theoretical range

        # Convert back to RGB
        with warnings.catch_warnings(record=True) as w: # Catch warnings to inspect them
            warnings.simplefilter("always") # Ensure all warnings are caught
            rgb = color.lab2rgb(denorm_lab) # Output range can be outside [0, 1]

            # Check for the specific "negative Z values" warning
            # for warn_item in w:
            #     if issubclass(warn_item.category, UserWarning) and "negative Z values" in str(warn_item.message):
            #         # Optionally print more info about the input lab_norm that caused this
            #         # print(f"DEBUG: lab_norm causing Z warning: min={np.min(lab_norm)}, max={np.max(lab_norm)}, mean={np.mean(lab_norm)}")
            #         pass


        # --- MODIFICATION: Handle potential NaNs from lab2rgb ---
        if np.any(np.isnan(rgb)):
            # This warning is critical as it means the conversion produced undefined values
            warnings.warn("NaNs detected in RGB output from lab2rgb (before final clipping). "
                          "Input denormalized LAB values might be severely out of sRGB gamut. Replacing NaNs with 0.0 (black).",
                          UserWarning)
            rgb = np.nan_to_num(rgb, nan=0.0) # Replace NaNs with 0.0 (black)
        # --- END MODIFICATION ---

        rgb = np.clip(rgb, 0.0, 1.0) # Ensure final RGB values are strictly within [0, 1]
        batch_rgb.append(rgb)

    rgb_batch = np.stack(batch_rgb, axis=0)
    if was_single_image:
        rgb_batch = rgb_batch[0]
    return rgb_batch.astype(np.float32)


# --- TensorFlow Wrappers (using tf.py_function) ---
@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
def tf_rgb_to_lab_normalized(rgb_image_tensor):
    lab_tensor = tf.py_function(
        func=rgb_to_lab_normalized,
        inp=[rgb_image_tensor],
        Tout=tf.float32
    )
    lab_tensor.set_shape([None, None, None, 3])
    return lab_tensor

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32)])
def tf_lab_normalized_to_rgb(lab_norm_tensor):
    rgb_tensor = tf.py_function(
        func=numpy_lab_normalized_to_rgb_clipped, # Calls the modified numpy function
        inp=[lab_norm_tensor],
        Tout=tf.float32
    )
    rgb_tensor.set_shape([None, None, None, 3])
    return rgb_tensor