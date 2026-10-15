
"""
Standalone NLP sentiment model
Project: Data-Driven Social Engagement Initiative

Trains a text classifier to label comments as:
- Relatable
- Neutral

Uses the project's synthetic demonstration dataset.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Locate the project directory and dataset
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "comments.csv"


def find_column(df, candidates, description):
    """Find a column using common possible column names."""
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find the {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def main():
    # 1. Load the dataset
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Check that comments.csv exists inside the data folder."
        )

    df = pd.read_csv(DATA_FILE)

    print("=" * 60)
    print("DATA-DRIVEN SOCIAL ENGAGEMENT INITIATIVE")
    print("Standalone NLP Sentiment Model")
    print("=" * 60)

    # 2. Identify the comment and label columns
    text_col = find_column(
        df,
        ["comment", "text", "comment_text", "content"],
        "comment text",
    )

    label_col = find_column(
        df,
        ["label", "sentiment", "category", "class"],
        "label",
    )

    # 3. Clean the data
    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()

    df = df[df[text_col] != ""]
    df = df[df[label_col] != ""]

    # Keep only the labels required by the project
    df[label_col] = df[label_col].str.title()
    df = df[df[label_col].isin(["Relatable", "Neutral"])]

    if len(df) < 4 or df[label_col].nunique() < 2:
        raise ValueError(
            "At least four labelled comments and both "
            "'Relatable' and 'Neutral' classes are needed."
        )

    print(f"\nDataset: {DATA_FILE.name}")
    print(f"Usable comments: {len(df)}")
    print("\nLabel distribution:")
    print(df[label_col].value_counts().to_string())

    # 4. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.25,
        random_state=42,
        stratify=df[label_col],
    )

    # 5. Convert text into numerical features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 6. Train the classification model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # 7. Evaluate the model on held-out comments
    predictions = model.predict(X_test_vec)

    print("\n" + "-" * 60)
    print("MODEL EVALUATION")
    print("-" * 60)
    print(f"Training comments: {len(X_train)}")
    print(f"Test comments: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["Neutral", "Relatable"],
            zero_division=0,
        )
    )

    # 8. Demonstrate predictions on example comments
    examples = [
        "I feel the same way.",
        "The post was uploaded on Monday.",
    ]

    print("\n" + "-" * 60)
    print("EXAMPLE PREDICTIONS")
    print("-" * 60)

    for comment in examples:
        features = vectorizer.transform([comment])
        predicted_label = model.predict(features)[0]
        print(f"Comment: {comment}")
        print(f"Predicted label: {predicted_label}\n")

    # 9. Interactive predictions
    print("Enter your own comment to classify.")
    print("Type 'exit' to finish.")

    while True:
        try:
            user_comment = input("\nComment: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram ended.")
            break

        if user_comment.lower() == "exit":
            print("Program ended.")
            break

        if not user_comment:
            print("Please enter a comment.")
            continue

        features = vectorizer.transform([user_comment])
        predicted_label = model.predict(features)[0]
        print(f"Predicted label: {predicted_label}")


if __name__ == "__main__":
    main()
