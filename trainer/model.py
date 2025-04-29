# trainer/model.py
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import sys

# ---> Import utils for LAB<->RGB conversion needed in compute_losses <---
from . import utils
from . import input as input_data # For image dimensions if needed

# Helper function (kept here as it's closely tied to the model architecture)
def concat(layers):
    return tf.concat(layers, axis=3)

# Main Model Class - Adapted from modelCRRNew.py
class ColorCastRemoval(tf.keras.Model):
    def __init__(self, decomnet_layer_num=5, **kwargs):
        super(ColorCastRemoval, self).__init__(**kwargs)
        self.DecomNet_layer_num = decomnet_layer_num

        # Initial conv layer - ensure it's used in call()
        self.initial_conv = tf.keras.layers.Conv2D(
            64, 3, padding='same', name="decom_initial_conv", activation=tf.nn.relu)
        
        # Residual and spatial convs as attributes
        self.residual_convs = []
        self.spatial_convs = []
        for i in range(self.DecomNet_layer_num):
            res_conv = tf.keras.layers.Conv2D(
                64, 3, padding='same', name=f"decom_res_conv_{i}", activation=tf.nn.relu)
            spat_conv = tf.keras.layers.Conv2D(
                1, 3, padding='same', name=f"decom_spat_conv_{i}", activation='sigmoid')
            self.residual_convs.append(res_conv)
            self.spatial_convs.append(spat_conv)
            # Explicitly track layers
            setattr(self, f'res_conv_{i}', res_conv)
            setattr(self, f'spat_conv_{i}', spat_conv)

        # Other layers
        self.channel_dense1 = tf.keras.layers.Dense(16, activation='relu', name="decom_chan_dense1")
        self.channel_dense2 = tf.keras.layers.Dense(64, activation='sigmoid', name="decom_chan_dense2")
        self.strength_conv = tf.keras.layers.Conv2D(2, 3, padding='same', name="decom_strength_conv", activation='sigmoid')
        self.offset_conv = tf.keras.layers.Conv2D(2, 3, padding='same', name="decom_offset_conv")
        self.illumination_conv_layer = tf.keras.layers.Conv2D(1, 3, padding='same', activation='linear', name="illum_conv")

        # VGG for perceptual loss
        vgg19 = VGG19(include_top=False, weights='imagenet')
        vgg19.trainable = False
        self.perceptual_loss_model = Model(
            inputs=vgg19.input,
            outputs=vgg19.get_layer('block4_conv2').output,
            name="vgg_perceptual"
        )
        self.perceptual_loss_model.trainable = False

        print(f"[*] ColorCastRemoval model initialized (Layers: {self.DecomNet_layer_num})")

        #print(f"[*] ColorCastRemoval model initialized (Layers: {self.DecomNet_layer_num}, VGG: '{perceptual_layer_name}')")
        # Store outputs as attributes for potential access (as in modelCRRNew.py)
        self.output_R = None
        self.output_R_corrected = None
        self.output_I = None
        self.channel_attention = None
        self.spatial_attention = None


    def get_config(self):
        base_config = super().get_config()
        config = {
            "decomnet_layer_num": self.DecomNet_layer_num,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def estimate_illumination(self, x):
        """Estimate the illumination map from the feature map."""
        # Use the layer defined in __init__
        illumination_map = self.illumination_conv_layer(x)
        # Normalize illumination map to [0, 1] using sigmoid
        illumination_map = tf.nn.sigmoid(illumination_map)
        return illumination_map

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        
        input_lum = inputs[:, :, :, 0:1]
        input_chroma = inputs[:, :, :, 1:3]

        # Initial conv - ensure output is used
        x = self.initial_conv(input_chroma)
        x = tf.nn.relu(x)

        # Residual blocks - ensure loop connects properly
        for i in range(self.DecomNet_layer_num):
            residual = self.residual_convs[i](x)
            
            # Channel attention
            squeeze = tf.reduce_mean(residual, axis=[1,2], keepdims=True)
            excitation = self.channel_dense1(squeeze)
            excitation = self.channel_dense2(excitation)
            residual = residual * excitation  # This multiplication must affect the final output

            # Spatial attention
            spatial_map_in = tf.reduce_mean(residual, axis=3, keepdims=True)
            spatial_attention_map = self.spatial_convs[i](spatial_map_in)
            residual = residual * spatial_attention_map  # This multiplication must affect the final output

            x += residual  # Critical connection point
            x = tf.nn.relu(x)

        # Final processing - ensure all these affect the output
        strength_mask = self.strength_conv(x)  # Must be used in corrected_chroma
        chroma_offset = self.offset_conv(x)     # Must be used in corrected_chroma
        corrected_chroma = input_chroma - (chroma_offset * strength_mask)  # Critical path
        # Inside model.py call method, after calculating offset/strength
        # tf.print("Strength Mask Stats:", tf.reduce_min(strength_mask), tf.reduce_max(strength_mask), tf.reduce_mean(strength_mask))
        # tf.print("Chroma Offset Stats:", tf.reduce_min(chroma_offset), tf.reduce_max(chroma_offset), tf.reduce_mean(chroma_offset))
        # # Or use tf.summary.histogram in training if preferred

        # Illumination estimation
        illumination_map = self.illumination_conv_layer(x)
        illumination_map = tf.nn.sigmoid(illumination_map)

        # Combine outputs
        output_R = tf.concat([input_lum, corrected_chroma], axis=-1)
        output_R_corrected = tf.clip_by_value(output_R, 0.0, 1.0)
        output_I = tf.clip_by_value(illumination_map, 0.0, 1.0)

        return output_R_corrected, output_I
        # --- Loss Calculation Methods (Adapted from modelCRRNew.py) ---

    def calculate_histogram_loss(self, pred_chroma, target_chroma):
        """Calculate histogram loss between predicted and target chroma (a,b channels)."""
        # Target can be neutral (0.5) or ground truth chroma
        scale = 1000.0
        scale_int32 = tf.cast(scale, tf.int32)
        value_range_int32 = [0, scale_int32]
        nbins = 50

        # Process 'a' channel (index 0 of chroma)
        pred_a_scaled = tf.cast(pred_chroma[..., 0] * scale, tf.int32)
        target_a_scaled = tf.cast(target_chroma[..., 0] * scale, tf.int32)
        hist_pred_a = tf.histogram_fixed_width(pred_a_scaled, value_range_int32, nbins=nbins)
        hist_target_a = tf.histogram_fixed_width(target_a_scaled, value_range_int32, nbins=nbins)

        # Process 'b' channel (index 1 of chroma)
        pred_b_scaled = tf.cast(pred_chroma[..., 1] * scale, tf.int32)
        target_b_scaled = tf.cast(target_chroma[..., 1] * scale, tf.int32)
        hist_pred_b = tf.histogram_fixed_width(pred_b_scaled, value_range_int32, nbins=nbins)
        hist_target_b = tf.histogram_fixed_width(target_b_scaled, value_range_int32, nbins=nbins)

        # Combine histograms
        hist_pred = tf.stack([hist_pred_a, hist_pred_b], axis=-1) # Shape [nbins, 2]
        hist_target = tf.stack([hist_target_a, hist_target_b], axis=-1)

        # Normalize and compute EMD (Earth Mover's Distance via CDF difference)
        hist_pred_float = tf.cast(hist_pred, tf.float32)
        hist_pred_norm = hist_pred_float / (tf.reduce_sum(hist_pred_float, axis=0, keepdims=True) + 1e-6) # Normalize per channel

        hist_target_float = tf.cast(hist_target, tf.float32)
        hist_target_norm = hist_target_float / (tf.reduce_sum(hist_target_float, axis=0, keepdims=True) + 1e-6) # Normalize per channel

        # Cumsum along the bin axis (axis 0)
        cum_pred = tf.cumsum(hist_pred_norm, axis=0)
        cum_target = tf.cumsum(hist_target_norm, axis=0)

        # Mean absolute difference of CDFs (EMD approximation) per channel
        loss_a = tf.reduce_mean(tf.abs(cum_pred[:, 0] - cum_target[:, 0]))
        loss_b = tf.reduce_mean(tf.abs(cum_pred[:, 1] - cum_target[:, 1]))

        return (loss_a + loss_b) / 2.0

    def detail_preservation_loss(self, original_lab, corrected_lab):
        """Focuses on preserving chrominance details (a/b channels)."""
        orig_ab = original_lab[..., 1:3] # Shape: [batch, H, W, 2]
        corr_ab = corrected_lab[..., 1:3] # Shape: [batch, H, W, 2]

        # Sobel edges for gradients - expects image format [batch, H, W, C]
        # We treat the 2 chroma channels as 'C'
        orig_grad = tf.image.sobel_edges(orig_ab) # Output shape: [batch, H, W, 2, 2] (last dim is y/x grad)
        corr_grad = tf.image.sobel_edges(corr_ab)

        # Calculate mean absolute difference over all gradient values
        return tf.reduce_mean(tf.abs(orig_grad - corr_grad))

    def calculate_perceptual_loss(self, output_rgb, target_rgb):
        """Calculates VGG perceptual loss between RGB images."""
        # Preprocess for VGG (expects RGB [0, 255])
        output_vgg = preprocess_input(output_rgb * 255.0)
        target_vgg = preprocess_input(target_rgb * 255.0)

        # Get features from the pre-trained VGG model
        output_features = self.perceptual_loss_model(output_vgg, training=False) # Ensure VGG is in inference mode
        target_features = self.perceptual_loss_model(target_vgg, training=False)

        # Calculate mean squared error between feature maps
        return tf.reduce_mean(tf.square(output_features - target_features))

    # --- compute_losses integrating logic from modelCRRNew.py ---
    # --- MODIFICATION: Add training argument ---
    def compute_losses(self, input_low_lab, input_high_lab, training=False):
        """
        Computes LAB-based losses.

        Args:
            input_low_lab: Low-quality input LAB image tensor.
            input_high_lab: High-quality target LAB image tensor.
            training: Boolean, passed to the model's call method.

        Returns:
            A tuple containing:
                - losses_dict: Dictionary of individual and total losses.
                - enhanced_lab: The corrected LAB output from the model.
        """
        # --- MODIFICATION: Pass training argument to self() ---
        # Forward pass - Use the training argument here
        enhanced_lab, illumination_map = self(input_low_lab, training=training)
        # --- END MODIFICATION ---

        # --- Calculate losses based on LAB ---
        # Reconstruction Loss (compare model output to target)
        recon_loss = tf.reduce_mean(tf.abs(enhanced_lab - input_high_lab))

        # Chroma Loss (compare output chroma to neutral gray 0.5)
        neutral_ab = tf.ones_like(enhanced_lab[..., 1:3]) * 0.5
        chroma_loss = tf.reduce_mean(tf.abs(enhanced_lab[..., 1:3] - neutral_ab))

        # Histogram Loss (compare output chroma histogram to neutral gray histogram)
        # Ensure this handles batches correctly if needed
        hist_loss = self.calculate_histogram_loss(enhanced_lab[..., 1:3], neutral_ab)

        # Detail Preservation Loss (compare output chroma gradients to *input* chroma gradients)
        detail_loss = self.detail_preservation_loss(input_low_lab, enhanced_lab)

        # Illumination Reconstruction Loss (consistency check)
        # Ensure denominator is safe from zero
        illum_recon_loss = tf.reduce_mean(tf.abs(
            input_low_lab[..., 0:1] - (enhanced_lab[..., 0:1] * illumination_map + 1e-6))) # Add epsilon

        # --- Weighted total loss ---
        # Adjust weights based on experimentation
        # Example weights (you might need different ones):
        # total_loss = (1.0 * recon_loss +
        #               0.1 * chroma_loss +  # Reduce weight if pushing too gray
        #               0.1 * hist_loss +    # Reduce weight if pushing too gray
        #               0.5 * detail_loss +  # Increase if details are lost
        #               0.3 * illum_recon_loss)
        # Original weights:
        total_loss = (1.0 * recon_loss +
                      0.7 * chroma_loss +
                      0.5 * hist_loss +
                      0.3 * detail_loss +
                      0.3 * illum_recon_loss)


        # --- Optional: Gradient printing for debugging ---
        # This requires persistent=True on the tape in train_step, which can use more memory.
        # Consider removing this once training is stable.
        # with tf.GradientTape(persistent=True) as tape_debug:
        #      tape_debug.watch(self.trainable_variables)
        #      # Recompute total_loss for gradient check if needed
        #      enhanced_lab_debug, illumination_map_debug = self(input_low_lab, training=training)
        #      # ... recompute losses ...
        #      total_loss_debug = ...
        # gradients_debug = tape_debug.gradient(total_loss_debug, self.trainable_variables)
        # for var, grad in zip(self.trainable_variables, gradients_debug):
        #      if grad is None:
        #          tf.print(f"DEBUG WARNING: No gradient for variable", var.name, output_stream=sys.stderr)
        #      else:
        #          tf.print(f"DEBUG Gradient OK for", var.name, "norm:", tf.norm(grad))
        # del tape_debug # Release persistent tape memory
        # --- End Optional Gradient Printing ---


        losses_dict = {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'chroma_loss': chroma_loss,
            'hist_loss': hist_loss,
            'detail_loss': detail_loss,
            'illum_recon_loss': illum_recon_loss
        }

        return losses_dict, enhanced_lab