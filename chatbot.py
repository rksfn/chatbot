import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim.downloader as api
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Download NLTK resources (run once per environment)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load lightweight pre-trained word2vec model
print("Loading word2vec model... This might take a moment")
word2vec_model = api.load("glove-wiki-gigaword-50")
print("Model loaded!")

# Intent Classification Dataset
data = [
    ("hi", "greeting"),
    ("hello there", "greeting"),
    ("hey", "greeting"),
    ("how are you", "question"),
    ("what's up", "question"),
    ("tell me about yourself", "question"),
    ("what's your name", "question_name"),
    ("who are you", "question_name"),
    ("tell me a joke", "joke"),
    ("say something funny", "joke"),
    ("cook something", "cooking"),
    ("how to bake", "cooking"),
]


# Reused Functions from Your Code
def get_pos_tag(word):
    """Convert NLTK POS tag to WordNet POS tag for lemmatization."""
    tag = nltk.pos_tag([word])[0][1]
    if tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def get_embedding(word):
    """Get the word2vec embedding for a word, if it exists."""
    try:
        return word2vec_model[word]
    except KeyError:
        return None


def get_synonyms(word, pos=wordnet.NOUN):
    """Get synonyms for a word using WordNet."""
    synonyms = set()
    for synset in wordnet.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().lower())
    return synonyms


def is_synonym_or_similar(
    word, target_words, pos=wordnet.NOUN, similarity_threshold=0.7
):
    """Check if a word is a synonym or semantically similar to any target word."""
    word_synonyms = get_synonyms(word, pos)
    for target in target_words:
        if word == target or target in word_synonyms:
            return True
        word_emb = get_embedding(word)
        if word_emb is not None:
            target_emb = get_embedding(target)
            if target_emb is not None:
                similarity = 1 - cosine(word_emb, target_emb)
                if similarity > similarity_threshold:
                    return True
    return False


# Train Intent Classifier
def train_intent_classifier(data):
    """Train a logistic regression model for intent classification."""
    # Extract sentences and labels using tuple indexing
    sentences = [row[0] for row in data]  # row[0] is the sentence
    labels = [row[1] for row in data]  # row[1] is the intent

    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(
        preprocessor=lambda x: " ".join(
            [
                lemmatizer.lemmatize(token, get_pos_tag(token))
                for token in word_tokenize(x.lower())
            ]
        )
    )
    X = vectorizer.fit_transform(sentences)
    y = labels

    # Train logistic regression model
    classifier = LogisticRegression(multi_class="ovr")
    classifier.fit(X, y)

    return vectorizer, classifier


# Train the model
vectorizer, classifier = train_intent_classifier(data)


# Updated Chatbot
def chatbot():
    print(
        "Hi! I'm an intent-based chatbot. Type 'exit' to quit or 'embed <word>' to see word embeddings."
    )
    while True:
        user_input = input("You: ").lower()
        tokens = word_tokenize(user_input)
        lemmatized_tokens = [
            lemmatizer.lemmatize(token, get_pos_tag(token)) for token in tokens
        ]
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
        else:
            # Predict intent using the classifier
            processed_input = " ".join(lemmatized_tokens)
            X_input = vectorizer.transform([processed_input])
            predicted_intent = classifier.predict(X_input)[0]

            # Respond based on predicted intent
            if predicted_intent == "greeting":
                print("Chatbot: Hey there! How can I help you?")
            elif predicted_intent == "question":
                print(
                    "Chatbot: I'm doing great, thanks for asking! What's on your mind?"
                )
            elif predicted_intent == "question_name":
                print("Chatbot: My name is GingBot, nice to meet you!")
            elif predicted_intent == "joke":
                print(
                    "Chatbot: Why did the scarecrow become a programmer? Because he was outstanding in his field!"
                )
            elif predicted_intent == "cooking":
                print(
                    "Chatbot: The veggie is better if cooked in the oven! That's a fact."
                )
            else:
                print(
                    "Chatbot: Hmm, I don't know that one. Try saying 'hello', 'how are you', 'tell me a joke', or 'cook something'."
                )


chatbot()
