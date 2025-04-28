# Choose an appropriate base image (e.g., Python 3.9)
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY ./trainer /app/trainer
COPY ./preprocess_images.py /app/preprocess_images.py 
# If needed by trainer/task.py
COPY ./evaluate_model.py /app/evaluate_model.py     
# If needed by trainer/task.py
# Add any other scripts needed by task.py

# Set the entrypoint to your main training script
ENTRYPOINT ["python", "-m", "trainer.task"]