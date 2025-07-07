#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please update it with your OpenAI API key."
fi

echo "Setup complete! Don't forget to:"
echo "1. Update .env with your OpenAI API key"
echo "2. Activate the virtual environment with: source venv/bin/activate" 