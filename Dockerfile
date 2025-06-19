# ---- Builder Stage ----
# Use the full Python image to build dependencies, as it has necessary build tools
FROM python:3.12.3 as builder

# Set the working directory
WORKDIR /app

# Install dependencies
# Using --no-cache-dir reduces layer size
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt


# ---- Final Stage ----
# Use the slim Python image for a much smaller final image
FROM python:3.12.3-slim

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy only the necessary application code and models
# Do NOT copy the training scripts or datasets
COPY models/ models/
COPY . .

# Define the command to run the API
CMD ["python", "api.py"]