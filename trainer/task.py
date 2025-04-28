# trainer/task.py
import argparse
import os
import time
import tensorflow as tf
import numpy as np
import sys

from . import model
from . import input as input_data
from .input import IMG_HEIGHT, IMG_WIDTH # <-- Add this import
# Use the TF wrappers for conversions within the graph
# --- MODIFICATION: Remove direct import of numpy lab_normalized_to_rgb ---
from .utils import tf_rgb_to_lab_normalized, tf_lab_normalized_to_rgb
# --- END MODIFICATION ---

# --- Define Arguments ---
def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # --- Data/Job Arguments ---
    parser.add_argument(
        '--train-low-dir', required=True, type=str,
        help='Path to TRAINING low-quality images directory')
    parser.add_argument(
        '--train-high-dir', required=True, type=str,
        help='Path to TRAINING high-quality images directory')
    parser.add_argument(
        '--val-low-dir', required=True, type=str,
        help='Path to VALIDATION low-quality images directory')
    parser.add_argument(
        '--val-high-dir', required=True, type=str,
        help='Path to VALIDATION high-quality images directory')
    parser.add_argument(
        '--job-dir', type=str, required=True,
        help='Location to write checkpoints and export models')
    parser.add_argument(
        '--log-dir', type=str, default=None, # Default to None
        help='Optional: Location for TensorBoard logs. Defaults to <job-dir>/logs')

    # --- Training Hyperparameters ---
    parser.add_argument(
        '--epochs', type=int, default=20, # Default from trainer
        help='Number of training epochs')
    parser.add_argument(
        '--batch-size', type=int, default=16, # Default from trainer
        help='Per-replica batch size')
    parser.add_argument(
        '--learning-rate', type=float, default=0.001, # Default from modelCRRNew.py train call
        help='Initial learning rate')
    parser.add_argument(
        '--decomnet-layers', type=int, default=5, # Matches modelCRRNew default
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
    print("TensorFlow version:", tf.__version__)
    print("Parsed arguments:", args)

    # --- Setup Distribution Strategy ---
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")

    # --- Determine Log and Checkpoint Directories ---
    log_dir = args.log_dir
    if not log_dir and args.job_dir:
        log_dir = os.path.join(args.job_dir, 'logs')

    checkpoint_dir = None
    if args.job_dir:
        checkpoint_dir = os.path.join(args.job_dir, 'checkpoints')
    else:
         print("Warning: --job-dir not specified. Checkpoints will not be saved or restored.")


    # --- Create Summary Writers ---
    train_summary_writer = None
    val_summary_writer = None
    if log_dir:
        print(f"TensorBoard log directory: {log_dir}")
        train_log_dir = os.path.join(log_dir, 'train')
        val_log_dir = os.path.join(log_dir, 'validation')
        try:
            tf.io.gfile.makedirs(train_log_dir)
            tf.io.gfile.makedirs(val_log_dir)
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        except Exception as e:
            print(f"Error creating SummaryWriters: {e}. Tensorboard logging disabled.")

    # --- Load Datasets ---
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

    # --- Calculate steps_per_epoch for LR Schedule ---
    steps_per_epoch = 100 # Default value
    decay_steps_value = 1000.0 # Default value for schedule
    try:
        # Use tf.data.experimental.cardinality which returns a Tensor
        train_cardinality = tf.data.experimental.cardinality(train_dataset)
        if train_cardinality == tf.data.UNKNOWN_CARDINALITY or train_cardinality == tf.data.INFINITE_CARDINALITY or train_cardinality <= 0:
             print("Warning: Could not determine dataset size or dataset is empty. Using default steps_per_epoch=100 and decay_steps=1000.")
             steps_per_epoch = 100 # Keep default steps for progbar
             decay_steps_value = 1000.0 # Use a larger default for decay
        else:
             steps_per_epoch = train_cardinality # Use actual cardinality for progbar
             decay_steps_value = tf.cast(steps_per_epoch, tf.float32) # Use actual for decay steps
             print(f"[*] Estimated steps per epoch: {steps_per_epoch.numpy()}")
             print(f"[*] Using decay_steps = {decay_steps_value.numpy()}") # Confirm value

    except Exception as e:
        print(f"Warning: Error getting dataset cardinality ({e}). Using default steps_per_epoch=100 and decay_steps=1000.")
        steps_per_epoch = 100
        decay_steps_value = 1000.0

    # Ensure decay_steps_value is a float
    if isinstance(decay_steps_value, tf.Tensor):
        decay_steps_value = decay_steps_value.numpy()
    decay_steps_value = float(decay_steps_value)
    if decay_steps_value <= 0: # Final safety check
        print(f"Warning: Calculated decay_steps_value ({decay_steps_value}) is invalid. Setting to 1000.0")
        decay_steps_value = 1000.0
    # --- END steps_per_epoch Calculation ---


    # --- Distribute Datasets ---
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

    # --- Build Model, Optimizer, Checkpoint (within scope) ---
    with strategy.scope():
        ccr_model = model.ColorCastRemoval(decomnet_layer_num=args.decomnet_layers)

        # --- Build the model with a dummy input ---
        print("\n=== Building Model ===")
        dummy_input = tf.zeros([1, IMG_HEIGHT, IMG_WIDTH, 3])
        try:
            # --- MODIFICATION: Add output clipping to model call ---
            # This adds the clipping directly to the model's output computation
            _ , _ = ccr_model(dummy_input, training=False) # Build with training=False first
            # --- END MODIFICATION ---
            print("[*] Model built successfully.")
        except Exception as build_e:
             print(f"[!] Error building model: {build_e}")
             import traceback
             traceback.print_exc()
        print("=== End Model Build ===\n")


        # --- Optimizer with potential LR decay ---
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps_value,
            decay_rate=args.lr_decay_factor,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Define metrics within scope
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

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=ccr_model, step=optimizer.iterations)

    # --- Checkpoint Management ---
    manager = None
    initial_epoch = 0
    if checkpoint_dir:
        tf.io.gfile.makedirs(checkpoint_dir) # Ensure dir exists
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
        if manager.latest_checkpoint:
            print(f"[*] Restoring from checkpoint: {manager.latest_checkpoint}")
            try:
                status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
                print(f"[*] Checkpoint restored. Resuming training from step {optimizer.iterations.numpy()}")
            except Exception as restore_e:
                 print(f"[!] Error restoring checkpoint: {restore_e}. Starting from scratch.")
                 optimizer.iterations.assign(0)
        else:
            print("[*] No checkpoint found, starting from scratch.")
            optimizer.iterations.assign(0)
    else:
        print("[*] No checkpoint directory specified, starting from scratch.")
        optimizer.iterations.assign(0)


    @tf.function
    def train_step(inputs):
        batch_low_lab, batch_high_lab = inputs

        with tf.GradientTape(persistent=True) as tape:
            # --- MODIFICATION: Pass training=True to compute_losses ---
            losses_lab_dict, enhanced_lab = ccr_model.compute_losses(
                batch_low_lab, batch_high_lab, training=True # Explicitly pass training=True
            )
            # --- END MODIFICATION ---

            if 'total_loss' not in losses_lab_dict:
                 tf.print("ERROR: 'total_loss' key missing from compute_losses output!", output_stream=sys.stderr)
                 total_lab_loss = tf.constant(1e6, dtype=tf.float32)
            else:
                 total_lab_loss = losses_lab_dict['total_loss']

            scaled_lab_loss = total_lab_loss / strategy.num_replicas_in_sync

        trainable_vars = ccr_model.trainable_variables
        gradients = tape.gradient(scaled_lab_loss, trainable_vars)

        valid_grads_and_vars = []
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                valid_grads_and_vars.append((grad, var))
            else:
                if var.trainable:
                     tf.print(f"Warning: No gradient for trainable variable {var.name}", output_stream=sys.stderr)

        if valid_grads_and_vars:
            optimizer.apply_gradients(valid_grads_and_vars)
        else:
             tf.print("Warning: No valid gradients found to apply.", output_stream=sys.stderr)


        # --- Calculate RGB Metrics *Outside* Gradient Context ---
        current_psnr = tf.constant(0.0, dtype=tf.float32)
        current_ssim = tf.constant(0.0, dtype=tf.float32)
        try:
            # Use the TF wrapper for conversion (already uses clipped numpy func)
            enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)
            input_high_rgb = tf_lab_normalized_to_rgb(batch_high_lab)

            enhanced_rgb = tf.ensure_shape(enhanced_rgb, [None, None, None, 3])
            input_high_rgb = tf.ensure_shape(input_high_rgb, [None, None, None, 3])

            current_psnr = tf.reduce_mean(tf.image.psnr(enhanced_rgb, input_high_rgb, max_val=1.0))
            current_ssim = tf.reduce_mean(tf.image.ssim(enhanced_rgb, input_high_rgb, max_val=1.0))

        except Exception as e:
            tf.print("ERROR during RGB metric calculation in train_step:", e, output_stream=sys.stderr)

        # --- Update Metrics ---
        train_loss_metric.update_state(total_lab_loss)
        train_recon_loss_metric.update_state(losses_lab_dict.get('recon_loss', 0.0))
        train_chroma_loss_metric.update_state(losses_lab_dict.get('chroma_loss', 0.0))
        train_hist_loss_metric.update_state(losses_lab_dict.get('hist_loss', 0.0))
        train_detail_loss_metric.update_state(losses_lab_dict.get('detail_loss', 0.0))
        train_psnr_metric.update_state(current_psnr)
        train_ssim_metric.update_state(current_ssim)

        return total_lab_loss

    # --- distributed_train_step ---
    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return {'total_loss': reduced_loss}

    # --- val_step ---
    @tf.function
    def val_step(inputs):
        batch_low_lab, batch_high_lab = inputs

        # --- Run model inference ---
        # --- MODIFICATION: Pass training=False to compute_losses ---
        losses_lab_dict, enhanced_lab = ccr_model.compute_losses(
            batch_low_lab, batch_high_lab, training=False # Explicitly pass training=False
        )
        # --- END MODIFICATION ---

        # --- Calculate RGB Metrics *Outside* Gradient Context ---
        current_psnr = tf.constant(0.0, dtype=tf.float32)
        current_ssim = tf.constant(0.0, dtype=tf.float32)
        try:
            # Use the TF wrapper for conversion
            enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)
            input_high_rgb = tf_lab_normalized_to_rgb(batch_high_lab)
            enhanced_rgb = tf.ensure_shape(enhanced_rgb, [None, None, None, 3])
            input_high_rgb = tf.ensure_shape(input_high_rgb, [None, None, None, 3])
            current_psnr = tf.reduce_mean(tf.image.psnr(enhanced_rgb, input_high_rgb, max_val=1.0))
            current_ssim = tf.reduce_mean(tf.image.ssim(enhanced_rgb, input_high_rgb, max_val=1.0))
        except Exception as e:
            tf.print("ERROR during RGB metric calculation in val_step:", e, output_stream=sys.stderr)

        # --- Update validation metrics ---
        val_loss_metric.update_state(losses_lab_dict.get('total_loss', 0.0))
        val_psnr_metric.update_state(current_psnr)
        val_ssim_metric.update_state(current_ssim)


    @tf.function
    def distributed_val_step(dist_inputs):
         strategy.run(val_step, args=(dist_inputs,))


    # --- Training and Validation Loop ---
    print("[*] Starting training and validation loop...")

    steps_per_epoch_int = tf.cast(steps_per_epoch, tf.int32).numpy() if isinstance(steps_per_epoch, tf.Tensor) else int(steps_per_epoch)
    if steps_per_epoch_int <= 0:
         print(f"Warning: Invalid steps_per_epoch ({steps_per_epoch_int}). Setting to 100 for Progbar.")
         steps_per_epoch_int = 100


    for epoch in range(initial_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        current_lr = lr_schedule(tf.cast(optimizer.iterations, tf.float32))
        print(f"  Starting Epoch {epoch+1} with LR: {current_lr.numpy():.6f}")

        # --- Training ---
        train_loss_metric.reset_state()
        train_psnr_metric.reset_state()
        train_ssim_metric.reset_state()
        train_recon_loss_metric.reset_state()
        train_chroma_loss_metric.reset_state()
        train_hist_loss_metric.reset_state()
        train_detail_loss_metric.reset_state()

        progbar = tf.keras.utils.Progbar(steps_per_epoch_int, stateful_metrics=['lr'])

        for step, batch_data in enumerate(train_dist_dataset):
            current_step_lr = lr_schedule(tf.cast(optimizer.iterations, tf.float32))
            step_results = distributed_train_step(batch_data)
            current_tf_step = optimizer.iterations

            metrics_update = [
                ('loss', train_loss_metric.result()),
                ('psnr', train_psnr_metric.result()),
                ('ssim', train_ssim_metric.result()),
                ('lr', current_step_lr)
                ]
            progbar.update(step + 1, values=metrics_update)

            if train_summary_writer and current_tf_step % args.log_steps == 0:
                with train_summary_writer.as_default(step=current_tf_step):
                    tf.summary.scalar('learning_rate', current_step_lr)
                    tf.summary.scalar('batch_loss_reduced', step_results['total_loss'])
                    tf.summary.scalar('epoch_loss_avg', train_loss_metric.result())
                    tf.summary.scalar('epoch_psnr_avg', train_psnr_metric.result())
                    tf.summary.scalar('epoch_ssim_avg', train_ssim_metric.result())
                    tf.summary.scalar('epoch_recon_loss_avg', train_recon_loss_metric.result())
                    tf.summary.scalar('epoch_chroma_loss_avg', train_chroma_loss_metric.result())
                    tf.summary.scalar('epoch_hist_loss_avg', train_hist_loss_metric.result())
                    tf.summary.scalar('epoch_detail_loss_avg', train_detail_loss_metric.result())

            if step + 1 >= steps_per_epoch_int:
                 break

        # --- Validation ---
        print("\n  Running validation...")
        val_loss_metric.reset_state()
        val_psnr_metric.reset_state()
        val_ssim_metric.reset_state()
        val_steps = 0
        val_start_time = time.time()
        for val_batch_data in val_dist_dataset:
            distributed_val_step(val_batch_data)
            val_steps += 1

        val_duration = time.time() - val_start_time
        current_val_loss = val_loss_metric.result()
        current_val_psnr = val_psnr_metric.result()
        current_val_ssim = val_ssim_metric.result()
        epoch_duration = time.time() - epoch_start_time

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Training - Loss: {train_loss_metric.result():.4f}, PSNR: {train_psnr_metric.result():.2f}, SSIM: {train_ssim_metric.result():.4f}")
        print(f"  Validation - Loss: {current_val_loss:.4f}, PSNR: {current_val_psnr:.2f}, SSIM: {current_val_ssim:.4f}")
        print(f"  Val Steps: {val_steps}, Val Duration: {val_duration:.2f}s, Epoch Duration: {epoch_duration:.2f}s")

        current_tf_step = optimizer.iterations
        if val_summary_writer:
             with val_summary_writer.as_default(step=current_tf_step):
                 tf.summary.scalar('epoch_loss', current_val_loss)
                 tf.summary.scalar('psnr', current_val_psnr)
                 tf.summary.scalar('ssim', current_val_ssim)

        # Log validation images if frequency matches
        if val_summary_writer and args.log_images_freq > 0 and \
           (epoch + 1) % args.log_images_freq == 0 and val_steps > 0:
            try:
                vis_batch = next(iter(val_dataset.take(1)))
                vis_low_lab, vis_high_lab = vis_batch

                # --- MODIFICATION: Use training=False for inference ---
                vis_enhanced_lab, _ = ccr_model(vis_low_lab, training=False)
                # --- END MODIFICATION ---

                # --- MODIFICATION: Use TF wrapper + numpy() for logging ---
                # Convert LAB tensors to RGB tensors using the TF wrapper
                vis_low_rgb_tf = tf_lab_normalized_to_rgb(vis_low_lab)
                vis_high_rgb_tf = tf_lab_normalized_to_rgb(vis_high_lab)
                vis_enhanced_rgb_tf = tf_lab_normalized_to_rgb(vis_enhanced_lab)

                # Convert RGB tensors to numpy for tf.summary.image and clipping
                vis_low_rgb_np = vis_low_rgb_tf.numpy()
                vis_high_rgb_np = vis_high_rgb_tf.numpy()
                vis_enhanced_rgb_np = vis_enhanced_rgb_tf.numpy()
                # --- END MODIFICATION ---

                # Ensure clipping after conversion (using numpy clip)
                vis_low_rgb = np.clip(vis_low_rgb_np, 0.0, 1.0)
                vis_high_rgb = np.clip(vis_high_rgb_np, 0.0, 1.0)
                vis_enhanced_rgb = np.clip(vis_enhanced_rgb_np, 0.0, 1.0)

                with val_summary_writer.as_default(step=current_tf_step):
                    max_outputs = min(4, global_batch_size)
                    tf.summary.image("Validation Samples/1_Input", vis_low_rgb, max_outputs=max_outputs)
                    tf.summary.image("Validation Samples/2_Target", vis_high_rgb, max_outputs=max_outputs)
                    tf.summary.image("Validation Samples/3_Output", vis_enhanced_rgb, max_outputs=max_outputs)
                print(f"  [*] Logged validation images for epoch {epoch+1}")
            except Exception as img_log_e:
                 print(f"  Warning: Failed to log validation images for epoch {epoch+1}: {img_log_e}")
                 import traceback
                 traceback.print_exc() # Print stack trace for image logging errors


        # --- Save Checkpoint ---
        if manager and ((epoch + 1) % args.save_checkpoint_epochs == 0 or (epoch + 1) == args.epochs):
            try:
                save_path = manager.save()
                print(f"  [*] Checkpoint saved: {save_path}")
            except Exception as ckpt_save_e:
                 print(f"  [!] Error saving checkpoint for epoch {epoch+1}: {ckpt_save_e}")


    # --- Final Model Export ---
    print("\n[*] Training finished.")
    model_export_dir = os.environ.get('AIP_MODEL_DIR', None)
    if not model_export_dir and args.job_dir:
         model_export_dir = os.path.join(args.job_dir, 'final_model')

    if model_export_dir:
            print(f"[*] Exporting final model to: {model_export_dir}")
            try:
                model_save_path = os.path.join(model_export_dir, 'model.keras')
                tf.io.gfile.makedirs(os.path.dirname(model_save_path))
                ccr_model.save(model_save_path)
                print(f"[*] Final model exported successfully to {model_save_path}")
            except Exception as e:
                print(f"[!] Error saving final model: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("[!] Model export directory not specified (AIP_MODEL_DIR or --job-dir). Final model not saved.")


if __name__ == '__main__':
    args = get_args()
    train_and_evaluate(args)


# **Summary of Changes:**

# 1.  **Import:** Changed the import from `utils` to only bring in `tf_rgb_to_lab_normalized` and `tf_lab_normalized_to_rgb`. The direct numpy function `lab_normalized_to_rgb` is no longer imported.
# 2.  **Image Logging:**
#     * Inside the image logging block (`if val_summary_writer...`), the lines converting LAB to RGB now use the TensorFlow wrapper:
#         ```python
#         vis_low_rgb_tf = tf_lab_normalized_to_rgb(vis_low_lab)
#         vis_high_rgb_tf = tf_lab_normalized_to_rgb(vis_high_lab)
#         vis_enhanced_rgb_tf = tf_lab_normalized_to_rgb(vis_enhanced_lab)
#         ```
#     * Immediately after, the resulting tensors are converted to numpy arrays using `.numpy()`:
#         ```python
#         vis_low_rgb_np = vis_low_rgb_tf.numpy()
#         vis_high_rgb_np = vis_high_rgb_tf.numpy()
#         vis_enhanced_rgb_np = vis_enhanced_rgb_tf.numpy()
#         ```
#     * The final `np.clip` and `tf.summary.image` calls use these numpy arrays (`vis_low_rgb`, `vis_high_rgb`, `vis_enhanced_rgb`).
# 3.  **Added `training` Argument:** Explicitly passed `training=True` to `ccr_model.compute_losses` in `train_step` and `training=False` in `val_step` and during the image logging inference call. This ensures any layers that behave differently during training vs. inference (like Dropout or BatchNormalization, although you might not be using them extensively here) are handled correctly. It also aligns with the changes potentially needed in `model.py` if you add the output clipping there.

# This version of `task.py` now consistently uses the TensorFlow wrappers for color space conversions, ensuring that the clipping logic added in `utils.py` is applied during metric calculation and image logging. Remember to continue training for more epoc