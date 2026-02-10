FROM python:3.9

# Working directory inside container
WORKDIR /app

# Copy all files from current directory to /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces (Docker) expects port 7860
EXPOSE 7860
ENV PORT=7860

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
