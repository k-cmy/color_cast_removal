# local_test.py
import sys
import os
import argparse # Import argparse to simulate command-line args

# --- Important: Add the project root to the Python path ---
# This allows Python to find the 'trainer' package when running this script from the root
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)
# ---------------------------------------------------------

from trainer.taskLOCAL import train_and_evaluate # Import your main function

if __name__ == '__main__':
    # --- Define Arguments Similar to Docker/Vertex AI ---
    # Create a namespace object to hold the arguments, mimicking argparse results
    args = argparse.Namespace()

    # --- Data/Job Arguments (Use LOCAL Paths) ---
    # Adjust these paths if your data is located elsewhere locally
    args.train_low_dir = os.path.join(project_root, 'data', 'low')
    args.train_high_dir = os.path.join(project_root, 'data', 'high')
    # Use separate validation data if available, otherwise use training data for quick test
    args.val_low_dir = os.path.join(project_root, 'data', 'low') # Or point to a validation set
    args.val_high_dir = os.path.join(project_root, 'data', 'high') # Or point to a validation set
    args.job_dir = os.path.join(project_root, 'local_output') # Base directory for local output
    # Checkpoints and logs will be placed inside job_dir by task.py logic if log_dir is None
    args.log_dir = None # Let task.py handle log dir creation inside job_dir

    # --- Training Hyperparameters (Use smaller values for quick tests) ---
    args.epochs = 20                 # Run only 1 epoch for a quick test
    args.batch_size = 2              # Use a small batch size
    args.learning_rate = 0.0001
    args.decomnet_layers = 7
    args.lr_decay_factor = 0.98      # <--- ADD THIS LINE (Match default in task.py)

    # --- Logging/Saving Arguments ---
    args.log_steps = 1               # Log every step for detailed view
    args.save_checkpoint_epochs = 1  # Save after the test epoch
    args.log_images_freq = 1         # Log images after the test epoch (if > 0)

    print("--- Starting Local Test ---")
    print(f"Arguments for local test: {args}")

    # --- Create local output directory if it doesn't exist ---
    # task.py now handles creating checkpoint/log dirs inside job_dir
    os.makedirs(args.job_dir, exist_ok=True)


    try:
        # --- Call your main training function ---
        train_and_evaluate(args)
        print("\n--- Local Test Completed Successfully ---")
    except Exception as e:
        print(f"\n--- Local Test Failed: {e} ---")
        # Optional: Re-raise the exception to get a full traceback
        import traceback
        traceback.print_exc()
        # raise e