FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the application code into the container at /app
COPY . .

# Command to run chatbot script
CMD ["python", "chatbot.py"]
