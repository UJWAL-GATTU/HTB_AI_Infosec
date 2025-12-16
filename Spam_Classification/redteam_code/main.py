import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def preprocess_message(message):
    stop_words = set(stopwords.words("english")) - {"free", "win", "cash", "urgent"}
    stemmer = PorterStemmer()

    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


def preprocess_dataframe(df):
    df['message'] = df['message'].apply(preprocess_message)
    df = df.drop_duplicates()

    return df

def classify_messages(model, msg_df, return_probabilities=False):
    if isinstance(msg_df, str):
        msg_preprocessed = [preprocess_message(msg_df)]
    else:
        msg_preprocessed = [preprocess_message(msg) for msg in msg_df]

    msg_vectorized = model.named_steps["vectorizer"].transform(msg_preprocessed)

    if return_probabilities:
        return model.named_steps["classifier"].predict_proba(msg_vectorized)

    return model.named_steps["classifier"].predict(msg_vectorized)



def train(dataset):
    df = pd.read_csv(dataset)
    df = preprocess_dataframe(df)
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["message"])
    y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", MultinomialNB())])
    param_grid = {"classifier__alpha": [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1")
    grid_search.fit(df["message"], y)
    best_model = grid_search.best_estimator_

    return best_model

def evaluate(model, dataset):
    df = pd.read_csv(dataset)
    df['label'] = df['label'].apply(lambda x: 1 if x == "spam" else 0)
    predictions = classify_messages(model, df['message'])
    correct = np.count_nonzero(predictions == df['label'])
    return (correct / len(df))

model = train("./train.csv")
acc = evaluate(model, "./test.csv")
print(f"Model accuracy: {round(acc*100, 2)}%")