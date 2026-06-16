import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

CONTRACTIONS = {
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "mustn't": "must not", "it's": "it is", "i'm": "i am",
    "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "they're": "they are",
    "they've": "they have", "we're": "we are", "we've": "we have",
    "that's": "that is", "there's": "there is", "what's": "what is",
    "let's": "let us", "who's": "who is", "how's": "how is"
}

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    nb    = joblib.load('model_nb.pkl')
    svm   = joblib.load('model_svm.pkl')
    return tfidf, nb, svm


def preprocess(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


# --- UI ---
st.set_page_config(page_title="Film Sentiment Classifier", layout="centered")
st.title("Film Review Sentiment Classifier")
st.caption("IMDB-based binary classification | Naive Bayes vs SVM")
st.divider()

tfidf, nb, svm = load_models()

review_input = st.text_area("Enter a movie review:", height=150,
                             placeholder="e.g. This movie was absolutely brilliant...")

if st.button("Classify", type="primary"):
    if not review_input.strip():
        st.warning("Review cannot be empty.")
    else:
        cleaned = preprocess(review_input)
        vec = tfidf.transform([cleaned])

        pred_nb  = nb.predict(vec)[0]
        pred_svm = svm.predict(vec)[0]

        label = {1: "Positive", 0: "Negative"}
        color = {1: "green", 0: "red"}

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Naive Bayes")
            st.markdown(
                f"<h2 style='color:{color[pred_nb]}'>{label[pred_nb]}</h2>",
                unsafe_allow_html=True
            )

        with col2:
            st.subheader("SVM")
            st.markdown(
                f"<h2 style='color:{color[pred_svm]}'>{label[pred_svm]}</h2>",
                unsafe_allow_html=True
            )

        with st.expander("Preprocessed text"):
            st.text(cleaned if cleaned else "(empty after preprocessing)")
