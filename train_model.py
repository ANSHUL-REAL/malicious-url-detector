import pandas as pd
import joblib
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def validate_url(url):
    """Returns True if the URL begins with http:// or https://"""
    return bool(re.match(r'^https?://', str(url)))


def preprocess_dataframe(df):
    """Filters valid URLs and standardizes the dataset"""
    df = df[df['url'].apply(validate_url)]
    df = df[['url', 'type']].dropna().drop_duplicates()
    df['type'] = df['type'].str.strip().str.lower()
    return df


def create_vectorizer():
    """Returns a TF-IDF vectorizer for character-level analysis"""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=1500,
        min_df=2,
        max_df=0.7,
        lowercase=True
    )


def train_model(X_train, y_train):
    """Trains a logistic regression model with preset parameters"""
    model = LogisticRegression(
        C=1,
        penalty='l2',
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Prints accuracy, report, and saves confusion matrix"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(y_true, y_pred):
    """Plots and saves the confusion matrix as a PNG file"""
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Malicious', 'Benign']
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")


def save_model_and_vectorizer(model, vectorizer):
    """Saves the model and vectorizer with timestamps"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, f"model_v{timestamp}.pkl")
    joblib.dump(vectorizer, f"vectorizer_v{timestamp}.pkl")
    print("Model and vectorizer saved.")


def main():
    try:
        print("Loading dataset...")
        df = pd.read_csv("malicious_urls.csv")

        print("Preprocessing data...")
        df = preprocess_dataframe(df)

        # Filter only supported labels
        df = df[df['type'].isin(['benign', 'phishing', 'malware', 'defacement'])]
        df['class'] = df['type'].apply(lambda x: 1 if x == 'benign' else -1)

        print("Label distribution:")
        print(df['class'].value_counts())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['url'], df['class'],
            test_size=0.2,
            random_state=42,
            stratify=df['class']
        )

        print("Vectorizing URLs...")
        vectorizer = create_vectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        print("Training model...")
        model = train_model(X_train_vec, y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test_vec, y_test)

        print("Saving model and vectorizer...")
        save_model_and_vectorizer(model, vectorizer)

        print("Training complete.")

    except Exception as error:
        print("An error occurred:")
        print(error)


if __name__ == "__main__":
    main()