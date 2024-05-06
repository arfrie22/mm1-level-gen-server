FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install Python dependencies
RUN apt-get update
RUN apt-get install -y git
RUN pip install git+https://github.com/crowsonkb/k-diffusion.git
RUN pip uninstall -y torch torchvision
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install python-dotenv asyncio aiohttp aiohttp_cors

# Copy the rest of your application code into the container
COPY . .

# Set the command to run your Python application
# You can change this command to match your application's entry point
CMD ["python", "server.py"]