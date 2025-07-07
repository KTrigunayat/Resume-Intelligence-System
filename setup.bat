@echo off

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install -r requirements.txt

REM Download spaCy model
python -m spacy download en_core_web_sm

REM Create .env file if it doesn't exist
if not exist .env (
    copy .env.example .env
    echo Created .env file. Please update it with your OpenAI API key.
)

echo Setup complete! Don't forget to:
echo 1. Update .env with your OpenAI API key
echo 2. Activate the virtual environment with: venv\Scripts\activate 