FROM gcr.io/google-appengine/python

# Set up any additional dependencies or configurations here

WORKDIR /app

# Copy your TensorFlow code into the container
COPY . /app

# Set the entry point for your TensorFlow script
ENTRYPOINT ["python", "build_model.py"]