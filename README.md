# Chatbot Project
A Python chatbot with NLP tokenization and word embeddings using NLTK and GloVe.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Download NLTK data: `python -c "import nltk; nltk.download('punkt_tab')"`
3. Run: `python chatbot.py`

## Features
- Tokenizes user input with NLTK.
- Shows word embeddings for input words using GloVe (first 5 dimensions).
- Responds to basic commands like "hello," "how are you," and "name."
