# trainer/model.py
from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, ReLU, LeakyReLU, BatchNormalization, GlobalAveragePooling2D, Multiply, Add, Concatenate, Activation
import sys

# ---> Import utils for LAB<->RGB conversion needed in compute_losses <---
from . import utils

class ColorCastRemoval(tf.keras.Model):
    def __init__(self, decomnet_layer_num=5, initial_filters=64, channel_attention_reduction=4, **kwargs):
        super(ColorCastRemoval, self).__init__(**kwargs)
        self.DecomNet_layer_num = decomnet_layer_num
        self.initial_filters = initial_filters
        self.channel_attention_reduction = channel_attention_reduction

        self.initial_conv = Conv2D(
            self.initial_filters, 3, padding='same', name="decom_initial_conv")
        self.initial_bn = BatchNormalization(name="decom_initial_bn")
        self.initial_relu = ReLU(name="decom_initial_relu")

        self.residual_convs = []
        self.residual_bns = []
        self.residual_relus = []

        self.channel_squeeze_pools = []
        self.channel_dense1_layers = []
        self.channel_relu_layers = []
        self.channel_dense2_layers = []
        self.channel_sigmoid_layers = []
        self.channel_multiply_layers = []

        self.spatial_attention_convs = []
        self.spatial_attention_sigmoids = []
        self.spatial_multiply_layers = []

        for i in range(self.DecomNet_layer_num):
            res_conv = Conv2D(
                self.initial_filters, 3, padding='same', name=f"decom_res_conv_{i}")
            self.residual_convs.append(res_conv)
            setattr(self, f'res_conv_{i}', res_conv)

            res_bn = BatchNormalization(name=f"decom_res_bn_{i}")
            self.residual_bns.append(res_bn)
            setattr(self, f'res_bn_{i}', res_bn)

            res_relu = ReLU(name=f"decom_res_relu_{i}")
            self.residual_relus.append(res_relu)
            setattr(self, f'res_relu_{i}', res_relu)

            self.channel_squeeze_pools.append(GlobalAveragePooling2D(keepdims=True, name=f"decom_ch_pool_{i}"))
            dense1 = Dense(self.initial_filters // self.channel_attention_reduction, name=f"decom_ch_dense1_{i}")
            self.channel_dense1_layers.append(dense1)
            setattr(self, f'ch_dense1_{i}', dense1)

            self.channel_relu_layers.append(ReLU(name=f"decom_ch_relu_{i}"))

            dense2 = Dense(self.initial_filters, name=f"decom_ch_dense2_{i}")
            self.channel_dense2_layers.append(dense2)
            setattr(self, f'ch_dense2_{i}', dense2)

            self.channel_sigmoid_layers.append(Activation('sigmoid', name=f"decom_ch_sigmoid_{i}"))
            self.channel_multiply_layers.append(Multiply(name=f"decom_ch_multiply_{i}"))

            spat_conv = Conv2D(
                1, 3, padding='same', name=f"decom_spat_conv_{i}")
            self.spatial_attention_convs.append(spat_conv)
            setattr(self, f'spat_att_conv_{i}', spat_conv)

            self.spatial_attention_sigmoids.append(Activation('sigmoid', name=f"decom_spat_sigmoid_{i}"))
            self.spatial_multiply_layers.append(Multiply(name=f"decom_spat_multiply_{i}"))

            setattr(self, f'add_{i}', Add(name=f"decom_add_{i}"))

        self.strength_conv = Conv2D(2, 3, padding='same', name="decom_strength_conv")
        self.strength_bn = BatchNormalization(name="decom_strength_bn")
        self.strength_sigmoid = Activation('sigmoid', name="decom_strength_sigmoid")

        self.offset_conv = Conv2D(2, 3, padding='same', name="decom_offset_conv")
        self.offset_bn = BatchNormalization(name="decom_offset_bn")

        self.illumination_conv_layer = Conv2D(1, 1, padding='same', activation='sigmoid', name="illum_conv")

        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(None, None, 3))
        vgg19.trainable = False
        self.perceptual_loss_model = Model(
            inputs=vgg19.input,
            outputs=vgg19.get_layer('block4_conv2').output,
            name="vgg_perceptual"
        )
        self.perceptual_loss_model.trainable = False

        print(f"[*] ColorCastRemoval model initialized (Layers: {self.DecomNet_layer_num}, Filters: {self.initial_filters}, VGG for Perceptual Loss)")

    def get_config(self):
        base_config = super().get_config()
        config = {
            "decomnet_layer_num": self.DecomNet_layer_num,
            "initial_filters": self.initial_filters,
            "channel_attention_reduction": self.channel_attention_reduction,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs_lab, training=False):
        inputs_lab = tf.cast(inputs_lab, tf.float32)

        input_lum = inputs_lab[:, :, :, 0:1]
        input_chroma = inputs_lab[:, :, :, 1:3]

        x = self.initial_conv(input_chroma)
        x = self.initial_bn(x, training=training)
        x = self.initial_relu(x)

        for i in range(self.DecomNet_layer_num):
            identity = x

            res_features = self.residual_convs[i](x)
            res_features = self.residual_bns[i](res_features, training=training)
            res_features = self.residual_relus[i](res_features)

            ch_squeeze = self.channel_squeeze_pools[i](res_features)
            ch_excite = self.channel_dense1_layers[i](ch_squeeze)
            ch_excite = self.channel_relu_layers[i](ch_excite)
            ch_excite = self.channel_dense2_layers[i](ch_excite)
            ch_attention_weights = self.channel_sigmoid_layers[i](ch_excite)
            attended_res_features = self.channel_multiply_layers[i]([res_features, ch_attention_weights])

            spatial_map_in = tf.reduce_mean(attended_res_features, axis=3, keepdims=True)
            spatial_attention_map = self.spatial_attention_convs[i](spatial_map_in)
            spatial_attention_weights = self.spatial_attention_sigmoids[i](spatial_attention_map)
            attended_spat_features = self.spatial_multiply_layers[i]([attended_res_features, spatial_attention_weights])

            x = getattr(self, f'add_{i}')([identity, attended_spat_features])

        strength_mask_ab = self.strength_conv(x)
        strength_mask_ab = self.strength_bn(strength_mask_ab, training=training)
        strength_mask_ab = self.strength_sigmoid(strength_mask_ab)

        offset_ab = self.offset_conv(x)
        offset_ab = self.offset_bn(offset_ab, training=training)

        corrected_chroma = input_chroma - (offset_ab * strength_mask_ab)

        illumination_map_L = self.illumination_conv_layer(x)

        output_R_lab = Concatenate(axis=-1, name="output_concat_lab")([input_lum, corrected_chroma])
        output_R_lab_corrected = tf.clip_by_value(output_R_lab, 0.0, 1.0, name="output_clip_lab")

        output_I = tf.clip_by_value(illumination_map_L, 0.0, 1.0, name="output_clip_illum")

        return output_R_lab_corrected, output_I

    def calculate_histogram_loss(self, pred_chroma, target_chroma):
        scale = 1000.0
        scale_int32 = tf.cast(scale, tf.int32)
        value_range_int32 = [0, scale_int32]
        nbins = 50

        pred_chroma = tf.cast(pred_chroma, tf.float32)
        target_chroma = tf.cast(target_chroma, tf.float32)

        losses_ab = []
        for i in range(2):
            pred_channel_scaled = tf.cast(pred_chroma[..., i] * scale, tf.int32)
            target_channel_scaled = tf.cast(target_chroma[..., i] * scale, tf.int32)

            hist_pred_int = tf.histogram_fixed_width(pred_channel_scaled, value_range_int32, nbins=nbins)
            hist_target_int = tf.histogram_fixed_width(target_channel_scaled, value_range_int32, nbins=nbins)

            hist_pred_float = tf.cast(hist_pred_int, tf.float32)
            hist_target_float = tf.cast(hist_target_int, tf.float32)

            hist_pred_norm = hist_pred_float / (tf.reduce_sum(hist_pred_float) + 1e-6)
            hist_target_norm = hist_target_float / (tf.reduce_sum(hist_target_float) + 1e-6)

            cum_pred = tf.cumsum(hist_pred_norm)
            cum_target = tf.cumsum(hist_target_norm)
            losses_ab.append(tf.reduce_mean(tf.abs(cum_pred - cum_target)))

        return (losses_ab[0] + losses_ab[1]) / 2.0

    def _calculate_sobel_loss_single_channel(self, original_sc, corrected_sc):
        # Helper for single-channel [B, H, W, 1] input
        original_sc = tf.cast(original_sc, tf.float32)
        corrected_sc = tf.cast(corrected_sc, tf.float32)
        orig_grad = tf.image.sobel_edges(original_sc)
        corr_grad = tf.image.sobel_edges(corrected_sc)
        return tf.reduce_mean(tf.abs(orig_grad - corr_grad))

    def detail_preservation_loss(self, original_lab_component, corrected_lab_component):
        # original_lab_component can be [B,H,W,1] (L) or [B,H,W,2] (AB)
        
        # Use static shape if available, as it's more robust for tf.function
        num_channels_static = original_lab_component.shape[-1]

        if num_channels_static == 1:
            return self._calculate_sobel_loss_single_channel(original_lab_component, corrected_lab_component)
        elif num_channels_static == 2:
            # Process A and B channels from the 2-channel input separately
            loss_a = self._calculate_sobel_loss_single_channel(
                original_lab_component[..., 0:1], 
                corrected_lab_component[..., 0:1]
            )
            loss_b = self._calculate_sobel_loss_single_channel(
                original_lab_component[..., 1:2], 
                corrected_lab_component[..., 1:2]
            )
            return (loss_a + loss_b) / 2.0
        else:
            # This path should ideally not be taken given the calls from compute_losses.
            # If num_channels_static is None (e.g., for a truly dynamic shape not known at graph construction),
            # then we would need to use tf.cond and tf.shape for dynamic dispatch.
            # For now, relying on the fact that compute_losses provides statically known channel sizes here.
            tf.print("Warning: detail_preservation_loss received component with unexpected static channel count:",
                     num_channels_static, "from shape:", original_lab_component.shape,
                     "Returning 0.0 for this loss component.",
                     output_stream=sys.stderr)
            return tf.constant(0.0, dtype=tf.float32)

    def calculate_perceptual_loss(self, output_rgb, target_rgb):
        output_rgb = tf.cast(output_rgb, tf.float32)
        target_rgb = tf.cast(target_rgb, tf.float32)

        output_vgg_input = preprocess_input(output_rgb * 255.0)
        target_vgg_input = preprocess_input(target_rgb * 255.0)

        output_features = self.perceptual_loss_model(output_vgg_input, training=False)
        target_features = self.perceptual_loss_model(target_vgg_input, training=False)

        return tf.reduce_mean(tf.square(output_features - target_features))

    def compute_losses(self, input_low_lab, input_high_lab, training=False):
        enhanced_lab, illumination_map = self(input_low_lab, training=training) # Pass training flag

        # --- DEBUG: Print enhanced_lab stats ---
        tf.print("enhanced_lab stats: min=", tf.reduce_min(enhanced_lab),
                 "max=", tf.reduce_max(enhanced_lab),
                 "mean=", tf.reduce_mean(enhanced_lab),
                 "has_nan=", tf.reduce_any(tf.math.is_nan(enhanced_lab)),
                 "has_inf=", tf.reduce_any(tf.math.is_inf(enhanced_lab)),
                 output_stream=sys.stderr)
        # --- END DEBUG ---

        try:
            enhanced_rgb = utils.tf_lab_normalized_to_rgb(enhanced_lab)
            input_high_rgb = utils.tf_lab_normalized_to_rgb(input_high_lab)
        except Exception as e:
            tf.print("Error during LAB->RGB conversion in compute_losses:", e, output_stream=sys.stderr)
            enhanced_rgb = tf.zeros_like(enhanced_lab, dtype=tf.float32)
            input_high_rgb = tf.zeros_like(input_high_lab, dtype=tf.float32)
        
        # --- DEBUG: Print converted RGB stats ---
        tf.print("enhanced_rgb stats (after lab2rgb): min=", tf.reduce_min(enhanced_rgb),
                 "max=", tf.reduce_max(enhanced_rgb),
                 "mean=", tf.reduce_mean(enhanced_rgb),
                 "has_nan=", tf.reduce_any(tf.math.is_nan(enhanced_rgb)),
                 "has_inf=", tf.reduce_any(tf.math.is_inf(enhanced_rgb)),
                 output_stream=sys.stderr)
        # --- END DEBUG ---


        recon_loss_lab = tf.reduce_mean(tf.abs(enhanced_lab - input_high_lab))
        chroma_recon_loss = tf.reduce_mean(tf.abs(enhanced_lab[..., 1:3] - input_high_lab[..., 1:3]))
        neutral_ab = tf.ones_like(enhanced_lab[..., 1:3]) * 0.5
        chroma_neutral_loss = tf.reduce_mean(tf.abs(enhanced_lab[..., 1:3] - neutral_ab))
        hist_loss_ab = self.calculate_histogram_loss(enhanced_lab[..., 1:3], input_high_lab[..., 1:3])
        detail_loss_L = self.detail_preservation_loss(input_low_lab[..., 0:1], enhanced_lab[..., 0:1])
        detail_loss_AB = self.detail_preservation_loss(input_low_lab[..., 1:3], enhanced_lab[..., 1:3])
        illum_recon_loss = tf.reduce_mean(tf.abs(
            input_low_lab[..., 0:1] - (enhanced_lab[..., 0:1] * illumination_map + 1e-6)))
        perceptual_loss_rgb = self.calculate_perceptual_loss(enhanced_rgb, input_high_rgb)
        tv_loss_enhanced_L = tf.reduce_mean(tf.image.total_variation(enhanced_lab[..., 0:1]))
        tv_loss_enhanced_AB = tf.reduce_mean(tf.image.total_variation(enhanced_lab[..., 1:3]))
        tv_loss_illum = tf.reduce_mean(tf.image.total_variation(illumination_map))

        # --- DEBUG: Print individual loss values ---
        tf.print("Individual losses: recon_lab=", recon_loss_lab,
                 "chroma_recon=", chroma_recon_loss, "chroma_neutral=", chroma_neutral_loss,
                 "hist_ab=", hist_loss_ab, "detail_L=", detail_loss_L, "detail_AB=", detail_loss_AB,
                 "illum_recon=", illum_recon_loss, "perceptual_rgb=", perceptual_loss_rgb,
                 "tv_L=", tv_loss_enhanced_L, "tv_AB=", tv_loss_enhanced_AB, "tv_illum=", tv_loss_illum,
                 output_stream=sys.stderr, summarize=-1)
        # --- END DEBUG ---

        total_loss = (
            1.0 * recon_loss_lab +
            1.0 * chroma_recon_loss +
            0.1 * chroma_neutral_loss +
            0.5 * hist_loss_ab +
            0.7 * detail_loss_L +
            0.5 * detail_loss_AB +
            0.3 * illum_recon_loss +
            0.05 * perceptual_loss_rgb + # Ensure perceptual_loss_rgb is not NaN
            0.001 * tv_loss_enhanced_L +
            0.001 * tv_loss_enhanced_AB +
            0.001 * tv_loss_illum
        )

        # --- DEBUG: Print total loss ---
        tf.print("total_loss=", total_loss,
                 "has_nan=", tf.reduce_any(tf.math.is_nan(total_loss)),
                 "has_inf=", tf.reduce_any(tf.math.is_inf(total_loss)),
                 output_stream=sys.stderr)
        # --- END DEBUG ---

        losses_dict = {
            'total_loss': total_loss,
            'recon_loss_lab': recon_loss_lab,
            'chroma_recon_loss': chroma_recon_loss,
            'chroma_neutral_loss': chroma_neutral_loss,
            'hist_loss_ab': hist_loss_ab,
            'detail_loss_L': detail_loss_L,
            'detail_loss_AB': detail_loss_AB,
            'illum_recon_loss': illum_recon_loss,
            'perceptual_loss_rgb': perceptual_loss_rgb,
            'tv_loss_L': tv_loss_enhanced_L,
            'tv_loss_AB': tv_loss_enhanced_AB,
            'tv_loss_illum': tv_loss_illum,
        }
        return losses_dict, enhanced_lab