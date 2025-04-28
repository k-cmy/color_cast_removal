# Use an official TensorFlow Serving image or a Python base image
# Option 1: Python base (more flexible, requires installing TF)
FROM python:3.9-slim

WORKDIR /app

# --- Install dependencies ---
# Copy only requirements first to leverage Docker cache
COPY color_cast_removal_trainer/requirements.txt requirements.txt
# Install system dependencies for OpenCV if needed (less common with headless)
# RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code ---
# Copy the trainer package (needed for model definition and utils)
COPY color_cast_removal_trainer/trainer ./trainer
# Copy the inference script
COPY predict.py ./predict.py
# Copy utility script if needed separately (though it's better inside trainer)
# COPY color_cast_removal_trainer/utils.py ./utils.py

# --- Copy Trained Model ---
# Assuming your trained model ('final_model/model.keras') is available
# Copy it into the image. Adjust the source path as needed.
# Example: If model is saved in 'output/final_model/model.keras' relative to Dockerfile context
COPY output/final_model/model.keras ./model.keras
# Or, if you prefer mounting the model at runtime:
# ENV MODEL_PATH=/app/model.keras

# --- Set Entrypoint/Command ---
# Run the prediction script
# CMD ["python", "predict.py", "--input_path", "path/to/input/image.png", "--output_path", "path/to/output/image.png"]
# Or, if building a prediction server (e.g., using Flask/FastAPI):
# EXPOSE 8080
# CMD ["python", "server.py"] # Assuming you create a server.py

ENTRYPOINT ["python", "predict.py"]
CMD ["--help"] # Default command if none provided