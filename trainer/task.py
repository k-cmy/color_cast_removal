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
from .utils import tf_rgb_to_lab_normalized, tf_lab_normalized_to_rgb

# --- Define Arguments ---
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
    print("TensorFlow version:", tf.__version__)
    print("Parsed arguments:", args)
    # Check if running on Vertex AI and print relevant env vars if they exist
    print(f"AIP_MODEL_DIR environment variable: {os.environ.get('AIP_MODEL_DIR')}")
    print(f"AIP_CHECKPOINT_DIR environment variable: {os.environ.get('AIP_CHECKPOINT_DIR')}")
    print(f"AIP_TENSORBOARD_LOG_DIR environment variable: {os.environ.get('AIP_TENSORBOARD_LOG_DIR')}")


    # --- Setup Distribution Strategy ---
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    global_batch_size = args.batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch_size}")

    # --- Determine Log and Checkpoint Directories ---
    # Vertex AI provides specific directories via env vars. Prioritize these if available.
    log_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR', None)
    if not log_dir:
        log_dir = args.log_dir # Use arg if AIP var not set
        if not log_dir and args.job_dir: # Fallback to job_dir if arg not set
             log_dir = os.path.join(args.job_dir, 'logs')
    print(f"Using log directory: {log_dir}")

    checkpoint_dir = os.environ.get('AIP_CHECKPOINT_DIR', None)
    if not checkpoint_dir and args.job_dir:
        # If not on Vertex AI (or AIP_CHECKPOINT_DIR not set), use job_dir for checkpoints
        checkpoint_dir = os.path.join(args.job_dir, 'checkpoints')
    print(f"Using checkpoint directory: {checkpoint_dir}")


    # --- Create Summary Writers (using tf.io.gfile for GCS compatibility) ---
    train_summary_writer = None
    val_summary_writer = None
    if log_dir:
        print(f"TensorBoard log directory specified: {log_dir}")
        train_log_dir = os.path.join(log_dir, 'train')
        val_log_dir = os.path.join(log_dir, 'validation')
        try:
            # Use tf.io.gfile.makedirs for compatibility with GCS
            tf.io.gfile.makedirs(train_log_dir)
            tf.io.gfile.makedirs(val_log_dir)
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            val_summary_writer = tf.summary.create_file_writer(val_log_dir)
            print(f"[*] TensorBoard logs will be written to: {log_dir}")
        except Exception as e:
            print(f"Error creating SummaryWriters in {log_dir}: {e}. Tensorboard logging disabled.")
            log_dir = None # Disable logging if writers failed
    else:
        print("[*] No log directory specified. Tensorboard logging disabled.")


    # --- Load Datasets (Paths come from args, can be GCS) ---
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
        train_cardinality = tf.data.experimental.cardinality(train_dataset)
        # Check for valid cardinality (tf.data.UNKNOWN_CARDINALITY is -1, tf.data.INFINITE_CARDINALITY is -2)
        if train_cardinality < 0 or train_cardinality == 0 :
            print(f"Warning: Could not determine training dataset size (cardinality={train_cardinality}). Using default steps_per_epoch=100 and decay_steps=1000.")
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

    # Ensure decay_steps_value is a float scalar
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
        dummy_input_shape = [1, IMG_HEIGHT, IMG_WIDTH, 3]
        print(f"Building model with dummy input shape: {dummy_input_shape}")
        dummy_input = tf.zeros(dummy_input_shape)
        try:
            # Build with training=False first
            _ , _ = ccr_model(dummy_input, training=False)
            print("[*] Model built successfully.")
            ccr_model.summary() # Print model summary
        except Exception as build_e:
            print(f"[!] Error building model: {build_e}")
            import traceback
            traceback.print_exc()
            sys.exit("Model build failed, exiting.") # Exit if model cannot be built
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

        # Checkpoint tracks optimizer state and model weights
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=ccr_model, step=optimizer.iterations)

    # --- Checkpoint Management (using tf.io.gfile for GCS compatibility) ---
    manager = None
    if checkpoint_dir:
        try:
            tf.io.gfile.makedirs(checkpoint_dir) # Ensure dir exists
            manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
            if manager.latest_checkpoint:
                print(f"[*] Attempting to restore from checkpoint: {manager.latest_checkpoint}")
                # Use expect_partial to handle potential changes in model structure gracefully
                status = checkpoint.restore(manager.latest_checkpoint).expect_partial()
                # status.assert_existing_objects_matched() # Optional: strict check
                print(f"[*] Checkpoint restored successfully. Resuming training from step {optimizer.iterations.numpy()}")
            else:
                print("[*] No checkpoint found in specified directory, starting from scratch.")
                optimizer.iterations.assign(0)
        except Exception as restore_e:
            print(f"[!] Error setting up CheckpointManager or restoring checkpoint: {restore_e}. Starting from scratch.")
            optimizer.iterations.assign(0)
            manager = None # Disable manager if setup failed
    else:
        print("[*] No checkpoint directory specified. Checkpoints will not be saved or restored.")
        optimizer.iterations.assign(0)


    # --- Training and Validation Steps (Defined as tf.function) ---

    @tf.function
    def train_step(inputs):
        batch_low_lab, batch_high_lab = inputs

        with tf.GradientTape(persistent=True) as tape:
            # Explicitly pass training=True
            losses_lab_dict, enhanced_lab = ccr_model.compute_losses(
                batch_low_lab, batch_high_lab, training=True
            )

            if 'total_loss' not in losses_lab_dict:
                tf.print("ERROR: 'total_loss' key missing from compute_losses output!", output_stream=sys.stderr)
                # Assign a default high loss or handle appropriately
                total_lab_loss = tf.constant(1e6, dtype=tf.float32) # Example default
            else:
                total_lab_loss = losses_lab_dict['total_loss']

            # Scale loss for distributed training BEFORE calculating gradients
            scaled_lab_loss = total_lab_loss / strategy.num_replicas_in_sync

        trainable_vars = ccr_model.trainable_variables
        gradients = tape.gradient(scaled_lab_loss, trainable_vars)

        # Filter out None gradients (can happen with disconnected parts of the graph)
        valid_grads_and_vars = []
        for grad, var in zip(gradients, trainable_vars):
            if grad is not None:
                valid_grads_and_vars.append((grad, var))
            else:
                # Optionally log variables that didn't get gradients
                # tf.print(f"Warning: No gradient for trainable variable {var.name}", output_stream=sys.stderr)
                pass # Avoid excessive logging

        if valid_grads_and_vars:
            optimizer.apply_gradients(valid_grads_and_vars)
        else:
             tf.print("Warning: No valid gradients found to apply for this step.", output_stream=sys.stderr)

        # --- Calculate RGB Metrics *Outside* Gradient Context ---
        current_psnr = tf.constant(0.0, dtype=tf.float32)
        current_ssim = tf.constant(0.0, dtype=tf.float32)
        try:
            enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)
            input_high_rgb = tf_lab_normalized_to_rgb(batch_high_lab)

            # Ensure shapes are defined for metrics (might be needed if shapes are partially known)
            enhanced_rgb = tf.ensure_shape(enhanced_rgb, [None, IMG_HEIGHT, IMG_WIDTH, 3])
            input_high_rgb = tf.ensure_shape(input_high_rgb, [None, IMG_HEIGHT, IMG_WIDTH, 3])

            # Use reduce_mean as PSNR/SSIM are calculated per image in batch
            current_psnr = tf.reduce_mean(tf.image.psnr(enhanced_rgb, input_high_rgb, max_val=1.0))
            current_ssim = tf.reduce_mean(tf.image.ssim(enhanced_rgb, input_high_rgb, max_val=1.0))

        except Exception as e:
            tf.print("ERROR during RGB metric calculation in train_step:", e, output_stream=sys.stderr)

        # --- Update Metrics ---
        train_loss_metric.update_state(total_lab_loss) # Use the unscaled loss for reporting
        train_recon_loss_metric.update_state(losses_lab_dict.get('recon_loss', 0.0))
        train_chroma_loss_metric.update_state(losses_lab_dict.get('chroma_loss', 0.0))
        train_hist_loss_metric.update_state(losses_lab_dict.get('hist_loss', 0.0))
        train_detail_loss_metric.update_state(losses_lab_dict.get('detail_loss', 0.0))
        train_psnr_metric.update_state(current_psnr)
        train_ssim_metric.update_state(current_ssim)

        return total_lab_loss # Return the unscaled loss

    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
        # Reduce the *unscaled* loss from each replica for reporting purposes
        reduced_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return {'total_loss': reduced_loss}


    @tf.function
    def val_step(inputs):
        batch_low_lab, batch_high_lab = inputs

        # --- Run model inference (training=False) ---
        losses_lab_dict, enhanced_lab = ccr_model.compute_losses(
            batch_low_lab, batch_high_lab, training=False # Explicitly pass training=False
        )

        # --- Calculate RGB Metrics ---
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

        # --- Update validation metrics ---
        val_loss_metric.update_state(losses_lab_dict.get('total_loss', 0.0))
        val_psnr_metric.update_state(current_psnr)
        val_ssim_metric.update_state(current_ssim)

        # Return images for potential logging
        return batch_low_lab, batch_high_lab, enhanced_lab

    @tf.function
    def distributed_val_step(dist_inputs):
        # Run the val_step on each replica
        # strategy.run returns values per replica, collect them if needed outside
        # For metric updates, strategy.run is sufficient as metrics aggregate across replicas
        return strategy.run(val_step, args=(dist_inputs,))


    # --- Training and Validation Loop ---
    print("[*] Starting training and validation loop...")

    # Ensure steps_per_epoch_int is valid for Progbar
    steps_per_epoch_int = int(steps_per_epoch.numpy()) if isinstance(steps_per_epoch, tf.Tensor) else int(steps_per_epoch)
    if steps_per_epoch_int <= 0:
         print(f"Warning: Invalid steps_per_epoch ({steps_per_epoch_int}). Setting to 100 for Progbar.")
         steps_per_epoch_int = 100

    # Calculate initial epoch based on restored step
    initial_epoch = 0
    if optimizer.iterations > 0 and steps_per_epoch_int > 0:
        initial_epoch = int(optimizer.iterations.numpy() // steps_per_epoch_int)
        print(f"Resuming from epoch {initial_epoch + 1}")


    for epoch in range(initial_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        # Get current learning rate for logging
        current_step_float = tf.cast(optimizer.iterations, tf.float32)
        current_lr = lr_schedule(current_step_float)
        print(f"  Starting Epoch {epoch+1} with LR: {current_lr.numpy():.6f}")

        # --- Training ---
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

            current_tf_step = optimizer.iterations # Get step *after* optimizer update

            # Update progress bar
            metrics_update = [
                ('loss', train_loss_metric.result()),
                ('psnr', train_psnr_metric.result()),
                ('ssim', train_ssim_metric.result()),
                ('lr', current_lr) # Show LR for the *start* of the epoch
                ]
            progbar.update(step + 1, values=metrics_update)

            # Log to TensorBoard periodically
            if train_summary_writer and current_tf_step.numpy() % args.log_steps == 0:
                with train_summary_writer.as_default(step=current_tf_step.numpy()):
                    tf.summary.scalar('learning_rate', lr_schedule(tf.cast(current_tf_step, tf.float32)))
                    tf.summary.scalar('batch_loss_reduced_mean', step_results['total_loss'])
                    tf.summary.scalar('epoch_loss_avg', train_loss_metric.result())
                    tf.summary.scalar('epoch_psnr_avg', train_psnr_metric.result())
                    tf.summary.scalar('epoch_ssim_avg', train_ssim_metric.result())
                    tf.summary.scalar('epoch_recon_loss_avg', train_recon_loss_metric.result())
                    tf.summary.scalar('epoch_chroma_loss_avg', train_chroma_loss_metric.result())
                    tf.summary.scalar('epoch_hist_loss_avg', train_hist_loss_metric.result())
                    tf.summary.scalar('epoch_detail_loss_avg', train_detail_loss_metric.result())
                    tf.summary.scalar('step_duration_ms', step_duration * 1000) # Log step time

            # Ensure loop breaks after steps_per_epoch iterations if dataset is repeating/infinite
            if step + 1 >= steps_per_epoch_int:
                break

        # --- Validation ---
        print("\n  Running validation...")
        val_loss_metric.reset_state()
        val_psnr_metric.reset_state()
        val_ssim_metric.reset_state()
        val_steps = 0
        first_val_batch_results = None # To store images for logging
        val_start_time = time.time()

        for val_batch_data in val_dist_dataset:
            # distributed_val_step runs val_step and updates metrics
            replica_results = distributed_val_step(val_batch_data)
            if first_val_batch_results is None:
                # Get results from the first replica for visualization
                # replica_results is a PerReplica object
                first_val_batch_results = (
                    replica_results[0][0], # batch_low_lab from replica 0
                    replica_results[0][1], # batch_high_lab from replica 0
                    replica_results[0][2]  # enhanced_lab from replica 0
                )
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

        current_tf_step_val = optimizer.iterations # Use current step for epoch summary
        if val_summary_writer:
            with val_summary_writer.as_default(step=current_tf_step_val.numpy()):
                tf.summary.scalar('epoch_loss', current_val_loss)
                tf.summary.scalar('epoch_psnr', current_val_psnr)
                tf.summary.scalar('epoch_ssim', current_val_ssim)

        # Log validation images if frequency matches and we have images
        if val_summary_writer and args.log_images_freq > 0 and \
           (epoch + 1) % args.log_images_freq == 0 and first_val_batch_results is not None:
            try:
                vis_low_lab, vis_high_lab, vis_enhanced_lab = first_val_batch_results

                # Convert LAB tensors to RGB tensors using the TF wrapper
                vis_low_rgb_tf = tf_lab_normalized_to_rgb(vis_low_lab)
                vis_high_rgb_tf = tf_lab_normalized_to_rgb(vis_high_lab)
                vis_enhanced_rgb_tf = tf_lab_normalized_to_rgb(vis_enhanced_lab)

                # Clipping is handled inside tf_lab_normalized_to_rgb via utils.lab_normalized_to_rgb
                # Ensure dtype is suitable for tf.summary.image (usually float32 in [0,1] or uint8)
                vis_low_rgb = tf.cast(vis_low_rgb_tf, tf.float32)
                vis_high_rgb = tf.cast(vis_high_rgb_tf, tf.float32)
                vis_enhanced_rgb = tf.cast(vis_enhanced_rgb_tf, tf.float32)

                with val_summary_writer.as_default(step=current_tf_step_val.numpy()):
                    max_outputs = min(4, vis_low_rgb.shape[0]) # Log up to 4 images from the first batch
                    tf.summary.image("Validation Samples/1_Input_LowQ", vis_low_rgb, max_outputs=max_outputs)
                    tf.summary.image("Validation Samples/2_Target_HighQ", vis_high_rgb, max_outputs=max_outputs)
                    tf.summary.image("Validation Samples/3_Output_Enhanced", vis_enhanced_rgb, max_outputs=max_outputs)
                print(f"  [*] Logged validation images for epoch {epoch+1}")
            except Exception as img_log_e:
                print(f"  Warning: Failed to log validation images for epoch {epoch+1}: {img_log_e}")
                # import traceback # Uncomment for detailed debugging
                # traceback.print_exc()


        # --- Save Checkpoint ---
        if manager and ((epoch + 1) % args.save_checkpoint_epochs == 0 or (epoch + 1) == args.epochs):
            try:
                save_path = manager.save()
                print(f"  [*] Checkpoint saved: {save_path}")
            except Exception as ckpt_save_e:
                print(f"  [!] Error saving checkpoint for epoch {epoch+1}: {ckpt_save_e}")

    # --- Final Model Export ---
    print("\n[*] Training finished.")

    # ---> MODIFICATION: Prioritize AIP_MODEL_DIR and save in SavedModel format <---
    # AIP_MODEL_DIR is automatically set by Vertex AI Training for the final model directory.
    model_export_dir = os.environ.get('AIP_MODEL_DIR', None)

    if not model_export_dir:
        # If AIP_MODEL_DIR is not set (e.g., local run), use job_dir as a fallback.
        if args.job_dir:
            model_export_dir = os.path.join(args.job_dir, 'final_model_export')
            print(f"[*] AIP_MODEL_DIR not set. Using fallback export directory: {model_export_dir}")
        else:
             print("[!] Model export directory not specified (AIP_MODEL_DIR or --job-dir fallback). Final model will not be saved.")

    if model_export_dir:
            print(f"[*] Exporting final model in SavedModel format to: {model_export_dir}")
            try:
                # Ensure the directory exists (important for GCS paths)
                # Note: model.save() often creates the directory, but explicit creation is safer.
                tf.io.gfile.makedirs(model_export_dir)

                # Save the model in TensorFlow SavedModel format (preferred for serving)
                # The serving signature might need adjustment based on how you plan to serve.
                # By default, it uses the model's call method.
                ccr_model.save(model_export_dir, save_format='tf')

                print(f"[*] Final model exported successfully to {model_export_dir}")

                # --- Optional: Add serving signature if needed for Vertex AI Prediction ---
                # This part is advanced and depends on your prediction input format.
                # Example: Define a serving function that takes e.g., base64 encoded images
                # @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
                # def serving_fn(encoded_image_bytes):
                #      # 1. Decode image bytes
                #      def decode_img(img_bytes):
                #          img = tf.image.decode_png(img_bytes, channels=3)
                #          img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
                #          img = tf.cast(img, tf.float32) / 255.0
                #          return img
                #      decoded_images = tf.map_fn(decode_img, encoded_image_bytes, dtype=tf.float32)
                #      # 2. Convert RGB to LAB
                #      input_lab = tf_rgb_to_lab_normalized(decoded_images)
                #      # 3. Run model inference (training=False)
                #      enhanced_lab, _ = ccr_model(input_lab, training=False)
                #      # 4. Convert LAB back to RGB
                #      enhanced_rgb = tf_lab_normalized_to_rgb(enhanced_lab)
                #      return {"output_image": enhanced_rgb} # Or desired output format
                #
                # print(f"[*] Saving model with custom serving signature 'serving_default'...")
                # tf.saved_model.save(ccr_model, model_export_dir, signatures={'serving_default': serving_fn})
                # print(f"[*] Model with custom signature saved to {model_export_dir}")
                # --- End Optional Signature ---


            except Exception as e:
                print(f"[!] Error saving final model to {model_export_dir}: {e}")
                import traceback
                traceback.print_exc()
    # ---> END MODIFICATION <---


if __name__ == '__main__':
    args = get_args()
    train_and_evaluate(args)