<img width="1920" height="1028" alt="image" src="https://github.com/user-attachments/assets/d8a036c1-952a-4bfd-9327-4b4070ca5fa0" /># Film Review Sentiment Classifier

Binary sentiment classification of movie reviews using Naive Bayes and SVM, with a Streamlit web interface for live prediction.

---

## Overview

This project trains two machine learning models to classify IMDB movie reviews as **Positive** or **Negative**. Both models are served through an interactive web app where users can type any review and instantly compare predictions from both classifiers.

---

## Dataset

- **Source:** IMDB Dataset
- **Size:** 50,000 reviews (25,000 positive / 25,000 negative)
- **Split:** 80/20 train-test with stratification
- **Review length:** 32 to 13,704 characters (mean ~1,309)

---

## Text Preprocessing

1. Remove HTML tags
2. Lowercase
3. Expand contractions (e.g. `isn't` -> `is not`)
4. Remove non-alphabetic characters
5. Tokenize
6. Remove stopwords
7. Porter stemming

---

## Models

| Model | Vectorizer | Features |
|---|---|---|
| Multinomial Naive Bayes | TF-IDF (unigrams + bigrams) | 50,000 |
| Linear SVC (SVM) | TF-IDF (unigrams + bigrams) | 50,000 |

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive Bayes | 87.92% | 0.8728 | 0.8878 | 0.8802 |
| **SVM (LinearSVC)** | **89.84%** | **0.8937** | **0.9044** | **0.8990** |

SVM outperforms Naive Bayes by ~2% across all metrics, with both models showing balanced precision and recall.

---

## Project Structure

```
Film-Review-Sentiment/
├── sentiment_classification.ipynb  # Training, evaluation, and analysis
├── app.py                          # Streamlit web app
├── model_nb.pkl                    # Trained Naive Bayes model
├── model_svm.pkl                   # Trained SVM model
├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
└── requirements.txt                # Python dependencies
```

---

## Installation

```bash
git clone https://github.com/Raxzell/Film-Review-Sentiment.git
cd Film-Review-Sentiment
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser, paste any movie review, and click **Classify** to see predictions from both models side by side.
**Or**
You can access it online without running it in your localhost by clicking this link: `https://film-review-sentiment-imdb.streamlit.app/`

---

## Dependencies

- `streamlit`
- `scikit-learn`
- `nltk`
- `joblib`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
