# Use a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the app files to the container
COPY main.py .
COPY data /app/data
COPY data /app/models
COPY notebooks /app/notebooks
COPY requirements.txt .

# Install the app dependencies
RUN pip install -r requirements.txt

# Expose the port that Streamlit will run on
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Set the command to run the Streamlit app
#CMD ["streamlit", "run", "main.py"]
