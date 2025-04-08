FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the NLTK data directory
ENV NLTK_DATA=/usr/local/share/nltk_data

# Download NLTK data directory
RUN  python -m nltk.downloader -d /usr/share/nltk_data all

# copy the rest of the application code into the container at /app
COPY . .

# Command to run chatbot script
CMD ["python", "chatbot.py"]

