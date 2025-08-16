import nltk
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import numpy as np

# Download punkt tokenizer (run once per environment)
nltk.download("punkt")
nltk.download("punkt_tab")

# Load lightweight pre-trained word2vec model
print("Loading word2vec model... This might take a moment")
word2vec_model = api.load("glove-wiki-gigaword-50")
print("Model loaded!")


def get_embedding(word):
    """Get the word2vec embedding for a word, if it exists."""
    try:
        return word2vec_model[word]
    except KeyError:
        return None


def chatbot():
    print(
        "Hi wanna chat? type 'exit' to quit or 'embed <word>' to see word embeddings."
    )
    while True:
        user_input = input("You: ").lower()
        tokens = word_tokenize(user_input)
        print(f"Chatbot: Your imput is tokenized: {tokens}")
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
        elif "hello" in user_input or "hi" in user_input:
            print("Chatbot: Hey there! How can I help you?")
        elif "how are you" in user_input:
            print("Chatbot: I'm doing great, thanks for asking!")
        elif "name" in user_input:
            print("Chatbot: My name is GingBot, nice to meet you!")
        else:
            print(
                "Chatbot: Hmm, I don't know that one. Try saying 'hello', 'how are you', or 'name'"
            )


chatbot()
