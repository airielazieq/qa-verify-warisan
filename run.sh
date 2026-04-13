#!/bin/bash
# Quick start guide for QA Cleaner with Ollama

# 1. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. Start Ollama (if not already running)
echo "Note: Make sure Ollama is running!"
echo "Run in another terminal: ollama serve"
echo ""

# 3. Pull the model (if not already pulled)
echo "Pulling Qwen 2.5 27B model..."
ollama pull qwen2.5:27b

# 4. Run on sample data
echo "Processing sample data with Ollama..."
python qa_cleaner.py sample_data.csv -o results.csv

# 5. Run on your own CSV file
# python qa_cleaner.py your_data.csv -o cleaned_results.csv --ollama-host http://localhost:11434 -m qwen2.5:27b

echo "Done! Check the output CSV file."

