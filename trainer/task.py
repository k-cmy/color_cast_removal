# trainer/task.py
import argparse
import os
import time
import tensorflow as tf
import numpy as np
import sys
import base64 # <--- Add import for base64 decoding (optional, but good practice for clarity)

from . import model
from . import input as input_data
# --- MODIFICATION: Import necessary items for serving_fn ---
from .input import IMG_HEIGHT, IMG_WIDTH
from .utils import tf_rgb_to_lab_normalized, tf_lab_normalized_to_rgb
# --- END MODIFICATION ---


# --- Define Arguments ---
# (get_args function remains the same as before)
def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # --- Data/Job Arguments ---
    parser.add_argument(
        '--train-low-dir', required=True, type=str,
        help='Path to TRAINING low-quality images directory (can be GCS path)')
    parser.add_argument(
        '--train-high-dir', required=True, type=str,
        help='Path to TRAINING high-quality images directory (can be GCS path)')
    parser.add_argument(
        '--val-low-dir', required=True, type=str,
        help='Path to VALIDATION low-quality images directory (can be GCS path)')
    parser.add_argument(
        '--val-high-dir', required=True, type=str,
        help='Path to VALIDATION high-quality images directory (can be GCS path)')
    parser.add_argument(
        '--job-dir', type=str, required=True,
        help='Base location for writing checkpoints and logs (can be GCS path). Also used for model export if AIP_MODEL_DIR is not set.') # Clarified help text
    parser.add_argument(
        '--log-dir', type=str, default=None, # Default to None
        help='Optional: Location for TensorBoard logs. Defaults to <job-dir>/logs (can be GCS path)')

    # --- Training Hyperparameters ---
    parser.add_argument(
        '--epochs', type=int, default=20,
        help='Number of training epochs')
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Per-replica batch size')
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Initial learning rate')
    parser.add_argument(
        '--decomnet-layers', type=int, default=5,
        help='Number of layers in the DecomNet-like blocks')
    parser.add_argument(
        '--lr-decay-factor', type=float, default=0.95,
        help='Learning rate decay factor per epoch (e.g., 0.95)')


    # --- Logging/Saving Arguments ---
    parser.add_argument(
        '--log-steps', type=int, default=100,
        help='Log training metrics every N global steps')
    parser.add_argument(
        '--save-checkpoint-epochs', type=int, default=1,
        help='Save checkpoint every N epochs')
    parser.add_argument(
        '--log-images-freq', type=int, default=1, # Log images every epoch by default
        help='Frequency (in epochs) to log sample validation images to TensorBoard. 0 disables.')

    return parser.parse_args()


# --- Main Training Function ---
def train_and_evaluate(args):
    """Runs the training and evaluation loop."""
    # --- (Initial setup: strategy, directories, datasets, etc. - remains the same) ---
    print("TensorFlow version:", tf.__version__)
    print("Parsed arguments:", args)
    print(f"AIP_MODEL_DIR environment variable: {os.environ.get('AIP_MODEL_DIR')}")
    print(f"AIP_CHECKPOINT_DIR environment variable: {os.environ.get('AIP_CHECKPOINT_DIR')}")
    print(f"AIP_TENSORBOARD_LOG_DIR environment variable: {os.environ.get('AIP_TENSORBOARD_LOG_DIR')}")

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")

    log_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR', None)
    if not log_dir:
        log_dir = args.log_dir
        if not log_dir and args.job_dir:
             log_dir = os.path.join(args.job_dir, 'logs')
    print(f"Using log directory: {log_dir}")

    checkpoint_dir = os.environ.get('AIP_CHECKPOINT_DIR', None)
    if not checkpoint_dir and args.job_dir:
        checkpoint_dir = os.path.join(args.job_dir, 'checkpoints')
    print(f"Using checkpoint directory: {checkpoint_dir}")

    train_summary_writer = None
    val_summary_writer = None
    if log_dir:
        print(f"TensorBoard log directory specified: {log_dir}")
        train_log_dir = os.path.join(log_dir, 'train')
        val_log_dir = os.path.join(log_dir, 'validation')
        try:
            tf.io.gfile.makedirs(train_log_dir)
            tf.io.gfile.makedirs(val_log_dir)
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            print(f"[*] TensorBoard logs will be written to: {log_dir}")
        except Exception as e:
            print(f"Error creating SummaryWriters in {log_dir}: {e}. Tensorboard logging disabled.")
            log_dir = None
    else:
        print("[*] No log directory specified. Tensorboard logging disabled.")

    print("Loading training dataset...")
    train_dataset = input_data.load_dataset(
        low_dir=args.train_low_dir, high_dir=args.train_high_dir,
        batch_size=global_batch_size, is_training=True
    )
    print("Loading validation dataset...")
    val_dataset = input_data.load_dataset(
        low_dir=args.val_low_dir, high_dir=args.val_high_dir,
        batch_size=global_batch_size, is_training=False
    )

    # Calculate steps_per_epoch, decay_steps_value (same as before)
    steps_per_epoch = 100
    decay_steps_value = 1000.0
    try:
        train_cardinality = tf.data.experimental.cardinality(train_dataset)
        if train_cardinality < 0 or train_cardinality == 0 :
            print(f"Warning: Could not determine training dataset size (cardinality={train_cardinality}). Using default steps_per_epoch=100 and decay_steps=1000.")
            steps_per_epoch = 100
            decay_steps_value = 1000.0
        else:
            steps_per_epoch = train_cardinality
            decay_steps_value = tf.cast(steps_per_epoch, tf.float32)
            print(f"[*] Estimated steps per epoch: {steps_per_epoch.numpy()}")
            print(f"[*] Using decay_steps = {decay_steps_value.numpy()}")

    except Exception as e:
        print(f"Warning: Error getting dataset cardinality ({e}). Using default steps_per_epoch=100 and decay_steps=1000.")
        steps_per_epoch = 100
        decay_steps_value = 1000.0

    if isinstance(decay_steps_value, tf.Tensor):
        decay_steps_value = decay_steps_value.numpy()
    decay_steps_value = float(decay_steps_value)
    if decay_steps_value <= 0:
        print(f"Warning: Calculated decay_steps_value ({decay_steps_value}) is invalid. Setting to 1000.0")
        decay_steps_value = 1000.0

    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # --- Build Model, Optimizer, Checkpoint (within scope) ---
    with strategy.scope():
        ccr_model = model.ColorCastRemoval(decomnet_layer_num=args.decomnet_layers)

        # Build the model
        print("\n=== Building Model ===")
        dummy_input_shape = [1, IMG_HEIGHT, IMG_WIDTH, 3]
        print(f"Building model with dummy input shape: {dummy_input_shape}")
        dummy_input = tf.zeros(dummy_input_shape)
        try:
            _ , _ = ccr_model(dummy_input, training=False)
            print("[*] Model built successfully.")
            # ccr_model.summary() # Optional: print model summary
        except Exception as build_e:
            print(f"[!] Error building model: {build_e}")
            import traceback
            traceback.print_exc()
            sys.exit("Model build failed, exiting.")
        print("=== End Model Build ===\n")

        # Optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps_value,
            decay_rate=args.lr_decay_factor,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Metrics
        train_loss_metric = tf.keras.metrics.Mean(name='train_total_loss')
        train_psnr_metric = tf.keras.metrics.Mean(name='train_psnr')
        train_ssim_metric = tf.keras.metrics.Mean(name='train_ssim')
        train_recon_loss_metric = tf.keras.metrics.Mean(name='train_recon_loss')
        train_chroma_loss_metric = tf.keras.metrics.Mean(name='train_chroma_loss')
        train_hist_loss_metric = tf.keras.metrics.Mean(name='train_hist_loss')
        train_detail_loss_metric = tf.keras.metrics.Mean(name='train_detail_loss')
        val_loss_metric = tf.keras.metrics.Mean(name='val_total_loss')
        val_psnr_metric = tf.keras.metrics.Mean(name='val_psnr')
        val_ssim_metric = tf.keras.metrics.Mean(name='val_ssim')

        # Checkpoint
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=ccr_model, step=optimizer.iterations)

    # Checkpoint Management (same as before)
    manager = None
    if checkpoint_dir:
        try:
            tf.io.gfile.makedirs(checkpoint_dir)
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
            if manager.latest_checkpoint:
                print(f"[*] Attempting to restore from checkpoint: {manager.latest_checkpoint}")
                status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
                print(f"[*] Checkpoint restored successfully. Resuming training from step {optimizer.iterations.numpy()}")
            else:
                print("[*] No checkpoint found in specified directory, starting from scratch.")
                optimizer.iterations.assign(0)
        except Exception as restore_e:
            print(f"[!] Error setting up CheckpointManager or restoring checkpoint: {restore_e}. Starting from scratch.")
            optimizer.iterations.assign(0)
            manager = None
    else:
        print("[*] No checkpoint directory specified. Checkpoints will not be saved or restored.")
        optimizer.iterations.assign(0)


    # --- Training and Validation Steps (Defined as tf.function) ---
    # (train_step, distributed_train_step, val_step, distributed_val_step remain the same)
    @tf.function
    def train_step(inputs):
        batch_low_lab, batch_high_lab = inputs
        with tf.GradientTape(persistent=True) as tape:
            losses_lab_dict, enhanced_lab = ccr_model.compute_losses(
                batch_low_lab, batch_high_lab, training=True
            )
            if 'total_loss' not in losses_lab_dict:
                tf.print("ERROR: 'total_loss' key missing from compute_losses output!", output_stream=sys.stderr)
                total_lab_loss = tf.constant(1e6, dtype=tf.float32)
            else:
                total_lab_loss = losses_lab_dict['total_loss']
            scaled_lab_loss = total_lab_loss / strategy.num_replicas_in_sync
        trainable_vars = ccr_model.trainable_variables
        gradients = tape.gradient(scaled_lab_loss, trainable_vars)
        valid_grads_and_vars = [(g, v) for g, v in zip(gradients, trainable_vars) if g is not None]
        if valid_grads_and_vars:
            optimizer.apply_gradients(valid_grads_and_vars)
        else:
             tf.print("Warning: No valid gradients found to apply for this step.", output_stream=sys.stderr)
        current_psnr = tf.constant(0.0, dtype=tf.float32)
        current_ssim = tf.constant(0.0, dtype=tf.float32)
        try:
            enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)
            input_high_rgb = tf_lab_normalized_to_rgb(batch_high_lab)
            enhanced_rgb = tf.ensure_shape(enhanced_rgb, [None, IMG_HEIGHT, IMG_WIDTH, 3])
            input_high_rgb = tf.ensure_shape(input_high_rgb, [None, IMG_HEIGHT, IMG_WIDTH, 3])
            current_psnr = tf.reduce_mean(tf.image.psnr(enhanced_rgb, input_high_rgb, max_val=1.0))
            current_ssim = tf.reduce_mean(tf.image.ssim(enhanced_rgb, input_high_rgb, max_val=1.0))
        except Exception as e:
            tf.print("ERROR during RGB metric calculation in train_step:", e, output_stream=sys.stderr)
        train_loss_metric.update_state(total_lab_loss)
        train_recon_loss_metric.update_state(losses_lab_dict.get('recon_loss', 0.0))
        train_chroma_loss_metric.update_state(losses_lab_dict.get('chroma_loss', 0.0))
        train_hist_loss_metric.update_state(losses_lab_dict.get('hist_loss', 0.0))
        train_detail_loss_metric.update_state(losses_lab_dict.get('detail_loss', 0.0))
        train_psnr_metric.update_state(current_psnr)
        train_ssim_metric.update_state(current_ssim)
        return total_lab_loss

    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return {'total_loss': reduced_loss}

    @tf.function
    def val_step(inputs):
        batch_low_lab, batch_high_lab = inputs
        losses_lab_dict, enhanced_lab = ccr_model.compute_losses(
            batch_low_lab, batch_high_lab, training=False
        )
        current_psnr = tf.constant(0.0, dtype=tf.float32)
        current_ssim = tf.constant(0.0, dtype=tf.float32)
        try:
            enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)
            input_high_rgb = tf_lab_normalized_to_rgb(batch_high_lab)
            enhanced_rgb = tf.ensure_shape(enhanced_rgb, [None, IMG_HEIGHT, IMG_WIDTH, 3])
            input_high_rgb = tf.ensure_shape(input_high_rgb, [None, IMG_HEIGHT, IMG_WIDTH, 3])
            current_psnr = tf.reduce_mean(tf.image.psnr(enhanced_rgb, input_high_rgb, max_val=1.0))
            current_ssim = tf.reduce_mean(tf.image.ssim(enhanced_rgb, input_high_rgb, max_val=1.0))
        except Exception as e:
            tf.print("ERROR during RGB metric calculation in val_step:", e, output_stream=sys.stderr)
        val_loss_metric.update_state(losses_lab_dict.get('total_loss', 0.0))
        val_psnr_metric.update_state(current_psnr)
        val_ssim_metric.update_state(current_ssim)
        return batch_low_lab, batch_high_lab, enhanced_lab # Return images for potential logging

    @tf.function
    def distributed_val_step(dist_inputs):
        return strategy.run(val_step, args=(dist_inputs,))


    # --- Training and Validation Loop ---
    # (Loop logic remains the same, including progress bar, logging, checkpoint saving)
    print("[*] Starting training and validation loop...")
    steps_per_epoch_int = int(steps_per_epoch.numpy()) if isinstance(steps_per_epoch, tf.Tensor) else int(steps_per_epoch)
    if steps_per_epoch_int <= 0:
         print(f"Warning: Invalid steps_per_epoch ({steps_per_epoch_int}). Setting to 100 for Progbar.")
         steps_per_epoch_int = 100
    initial_epoch = 0
    if optimizer.iterations > 0 and steps_per_epoch_int > 0:
        initial_epoch = int(optimizer.iterations.numpy() // steps_per_epoch_int)
        print(f"Resuming from epoch {initial_epoch + 1}")

    for epoch in range(initial_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        current_step_float = tf.cast(optimizer.iterations, tf.float32)
        current_lr = lr_schedule(current_step_float)
        print(f"  Starting Epoch {epoch+1} with LR: {current_lr.numpy():.6f}")

        # Training Phase
        train_loss_metric.reset_state()
        train_psnr_metric.reset_state()
        train_ssim_metric.reset_state()
        train_recon_loss_metric.reset_state()
        train_chroma_loss_metric.reset_state()
        train_hist_loss_metric.reset_state()
        train_detail_loss_metric.reset_state()
        progbar = tf.keras.utils.Progbar(steps_per_epoch_int, stateful_metrics=['lr', 'loss', 'psnr', 'ssim'])
        for step, batch_data in enumerate(train_dist_dataset):
            step_start_time = time.time()
            step_results = distributed_train_step(batch_data)
            step_duration = time.time() - step_start_time
            current_tf_step = optimizer.iterations
            metrics_update = [('loss', train_loss_metric.result()), ('psnr', train_psnr_metric.result()),
                              ('ssim', train_ssim_metric.result()), ('lr', current_lr)]
            progbar.update(step + 1, values=metrics_update)
            if train_summary_writer and current_tf_step.numpy() % args.log_steps == 0:
                with train_summary_writer.as_default(step=current_tf_step.numpy()):
                    tf.summary.scalar('learning_rate', lr_schedule(tf.cast(current_tf_step, tf.float32)))
                    tf.summary.scalar('batch_loss_reduced_mean', step_results['total_loss'])
                    tf.summary.scalar('epoch_loss_avg', train_loss_metric.result())
                    tf.summary.scalar('epoch_psnr_avg', train_psnr_metric.result())
                    # ... (other train metrics) ...
                    tf.summary.scalar('step_duration_ms', step_duration * 1000)
            if step + 1 >= steps_per_epoch_int: break

        # Validation Phase
        print("\n  Running validation...")
        val_loss_metric.reset_state()
        val_psnr_metric.reset_state()
        val_ssim_metric.reset_state()
        val_steps = 0
        first_val_batch_results = None
        val_start_time = time.time()
        for val_batch_data in val_dist_dataset:
            replica_results = distributed_val_step(val_batch_data)
            if first_val_batch_results is None and replica_results: # Ensure replica_results is not empty
                 # Safely access results from the first replica if available
                 try:
                     first_val_batch_results = (
                         replica_results[0][0], # batch_low_lab from replica 0
                         replica_results[0][1], # batch_high_lab from replica 0
                         replica_results[0][2]  # enhanced_lab from replica 0
                     )
                 except (IndexError, TypeError):
                     print("Warning: Could not retrieve validation batch results for image logging.")
                     first_val_batch_results = None # Reset if access failed
            val_steps += 1
        val_duration = time.time() - val_start_time
        current_val_loss = val_loss_metric.result()
        current_val_psnr = val_psnr_metric.result()
        current_val_ssim = val_ssim_metric.result()
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training   - Loss: {train_loss_metric.result():.4f}, PSNR: {train_psnr_metric.result():.2f}, SSIM: {train_ssim_metric.result():.4f}")
        print(f"  Validation - Loss: {current_val_loss:.4f}, PSNR: {current_val_psnr:.2f}, SSIM: {current_val_ssim:.4f}")
        print(f"  Val Steps: {val_steps}, Val Duration: {val_duration:.2f}s, Epoch Duration: {epoch_duration:.2f}s")

        current_tf_step_val = optimizer.iterations
        if val_summary_writer:
            with val_summary_writer.as_default(step=current_tf_step_val.numpy()):
                tf.summary.scalar('epoch_loss', current_val_loss)
                tf.summary.scalar('epoch_psnr', current_val_psnr)
                tf.summary.scalar('epoch_ssim', current_val_ssim)

        # Log validation images
        if val_summary_writer and args.log_images_freq > 0 and \
           (epoch + 1) % args.log_images_freq == 0 and first_val_batch_results is not None:
            try:
                vis_low_lab, vis_high_lab, vis_enhanced_lab = first_val_batch_results
                vis_low_rgb_tf = tf_lab_normalized_to_rgb(vis_low_lab)
                vis_high_rgb_tf = tf_lab_normalized_to_rgb(vis_high_lab)
                vis_enhanced_rgb_tf = tf_lab_normalized_to_rgb(vis_enhanced_lab)
                vis_low_rgb = tf.cast(vis_low_rgb_tf, tf.float32)
                vis_high_rgb = tf.cast(vis_high_rgb_tf, tf.float32)
                vis_enhanced_rgb = tf.cast(vis_enhanced_rgb_tf, tf.float32)
                with val_summary_writer.as_default(step=current_tf_step_val.numpy()):
                    max_outputs = min(4, vis_low_rgb.shape[0])
                    tf.summary.image("Validation Samples/1_Input_LowQ", vis_low_rgb, max_outputs=max_outputs)
                    tf.summary.image("Validation Samples/2_Target_HighQ", vis_high_rgb, max_outputs=max_outputs)
                    tf.summary.image("Validation Samples/3_Output_Enhanced", vis_enhanced_rgb, max_outputs=max_outputs)
                print(f"  [*] Logged validation images for epoch {epoch+1}")
            except Exception as img_log_e:
                print(f"  Warning: Failed to log validation images for epoch {epoch+1}: {img_log_e}")

        # Save Checkpoint
        if manager and ((epoch + 1) % args.save_checkpoint_epochs == 0 or (epoch + 1) == args.epochs):
            try:
                save_path = manager.save()
                print(f"  [*] Checkpoint saved: {save_path}")
            except Exception as ckpt_save_e:
                print(f"  [!] Error saving checkpoint for epoch {epoch+1}: {ckpt_save_e}")


    # --- Final Model Export ---
    print("\n[*] Training finished.")

    # ---> MODIFICATION: Define the Serving Function <---
    print("[*] Defining serving function for SavedModel...")

    # This function processes a single image (base64 string -> RGB tensor)
    # It will be used with tf.map_fn to handle batches.
    def _preprocess_image(encoded_string):
        # 1. Decode Base64
        img_bytes = tf.io.decode_base64(encoded_string)
        # 2. Decode PNG (assuming PNG input)
        # Use tf.image.decode_image if you need to support multiple formats (JPG, PNG, GIF, BMP)
        img_rgb_uint8 = tf.image.decode_png(img_bytes, channels=3)
         # Set shape explicitly after decoding if needed, though resize will handle it
        img_rgb_uint8.set_shape([None, None, 3])
        # 3. Resize
        img_resized = tf.image.resize(img_rgb_uint8, [IMG_HEIGHT, IMG_WIDTH])
        # 4. Cast and Normalize RGB
        img_rgb_float = tf.cast(img_resized, tf.float32) / 255.0
        # 5. Clip RGB values
        img_rgb_clipped = tf.clip_by_value(img_rgb_float, 0.0, 1.0)
        # 6. Convert RGB to Normalized LAB
        # tf_rgb_to_lab_normalized expects a batch dim, so add it temporarily
        img_lab = tf_rgb_to_lab_normalized(tf.expand_dims(img_rgb_clipped, axis=0))
        # Remove the batch dim added for the conversion function
        return tf.squeeze(img_lab, axis=0)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serving_fn(image_bytes_list):
        """
        Accepts a batch of base64 encoded PNG image strings, processes them,
        runs inference, and returns the enhanced RGB images.
        """
        print("Executing serving_fn") # Add a print for debugging/confirmation

        # Use tf.map_fn to apply preprocessing to each image in the batch
        # This maps _preprocess_image over the batch of encoded strings
        input_lab_batch = tf.map_fn(
            _preprocess_image, image_bytes_list, dtype=tf.float32,
            fn_output_signature=tf.TensorSpec(shape=[IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
        )
        input_lab_batch = tf.ensure_shape(input_lab_batch, [None, IMG_HEIGHT, IMG_WIDTH, 3])

        tf.print("Serving function: Input LAB shape:", tf.shape(input_lab_batch))

        # Run the core model inference (training=False is default but explicit is good)
        # Your model's call method returns (output_R_corrected, output_I)
        enhanced_lab, _ = ccr_model(input_lab_batch, training=False)

        tf.print("Serving function: Enhanced LAB shape:", tf.shape(enhanced_lab))

        # Convert enhanced LAB output back to RGB
        enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)

        # Ensure clipping after conversion (safety check, may be redundant if done in utils)
        enhanced_rgb_clipped = tf.clip_by_value(enhanced_rgb, 0.0, 1.0)

        tf.print("Serving function: Output RGB shape:", tf.shape(enhanced_rgb_clipped))

        # Return the result in a dictionary format (standard for TF Serving / Vertex AI)
        return {"enhanced_image": enhanced_rgb_clipped}

    # ---> END Define Serving Function <---


    # Determine export directory (Prioritize AIP_MODEL_DIR)
    model_export_dir = os.environ.get('AIP_MODEL_DIR', None)
    if not model_export_dir:
        if args.job_dir:
            model_export_dir = os.path.join(args.job_dir, 'final_model_export')
            print(f"[*] AIP_MODEL_DIR not set. Using fallback export directory: {model_export_dir}")
        else:
             print("[!] Model export directory not specified (AIP_MODEL_DIR or --job-dir fallback). Final model will not be saved.")

    if model_export_dir:
            # ---> MODIFICATION: Save with the custom signature <---
            print(f"[*] Exporting final model with 'serving_default' signature to: {model_export_dir}")
            try:
                tf.io.gfile.makedirs(model_export_dir) # Ensure directory exists

                # Save the model using tf.saved_model.save, providing the model,
                # the directory, and the desired signature definition.
                tf.saved_model.save(
                    ccr_model,
                    model_export_dir,
                    signatures={'serving_default': serving_fn} # Map 'serving_default' to our serving_fn
                )

                print(f"[*] Final model with serving signature exported successfully to {model_export_dir}")

            except Exception as e:
                print(f"[!] Error saving final model with signature to {model_export_dir}: {e}")
                import traceback
                traceback.print_exc()
            # ---> END Save Modification <---


if __name__ == '__main__':
    args = get_args()
    train_and_evaluate(args)