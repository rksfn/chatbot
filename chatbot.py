import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim.downloader as api
import numpy as np

# Download NLTK resources (run once per environment)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")  # Added for POS tagging

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load lightweight pre-trained word2vec model
print("Loading word2vec model... This might take a moment")
word2vec_model = api.load("glove-wiki-gigaword-50")
print("Model loaded!")

def get_pos_tag(word):
    """Convert NLTK POS tag to WordNet POS tag for lemmatization."""
    tag = nltk.pos_tag([word])[0][1]
    if tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN  # Default to noun if unknown

def get_embedding(word):
    """Get the word2vec embedding for a word, if it exists."""
    try:
        return word2vec_model[word]
    except KeyError:
        return None

def chatbot():
    print(
        "Hi wanna chat? Type 'exit' to quit or 'embed <word>' to see word embeddings."
    )
    while True:
        user_input = input("You: ").lower()
        # Tokenize and lemmatize input with POS tagging
        tokens = word_tokenize(user_input)
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_pos_tag(token)) for token in tokens]
        print(f"Chatbot: Your input is tokenized: {tokens}")
        print(f"Chatbot: Lemmatized tokens: {lemmatized_tokens}")
        
        if user_input == "exit":
            print("Bye!")
            break
        elif user_input.startswith("embed "):
            word = user_input[6:].strip()
            embedding = get_embedding(word)
            if embedding is not None:
                print(
                    f"Chatbot: Embedding for '{word}' (first 5 dimensions): {embedding[:5]}"
                )
            else:
                print(f"Chatbot: Sorry, I don't have embedding for '{word}'.")
        elif any(lemma in ["hello", "hi"] for lemma in lemmatized_tokens):
            print("Chatbot: Hey there! How can I help you?")
        elif any(lemma == "you" for lemma in lemmatized_tokens):
            print("Chatbot: I'm doing great, thanks for asking!")
        elif any(lemma == "name" for lemma in lemmatized_tokens):
            print("Chatbot: My name is GingBot, nice to meet you!")
        else:
            print(
                "Chatbot: Hmm, I don't know that one. Try saying 'hello', 'how are you', or 'name'"
            )

chatbot()
