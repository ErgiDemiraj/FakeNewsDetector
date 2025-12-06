"""
Training script for the fake news detector.
Uses the Kaggle fake news dataset (Fake.csv and True.csv).
"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

print("Loading datasets...")

# Load the fake and true news datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels: 1 = fake, 0 = real
fake_df['label'] = 1
true_df['label'] = 0

# Combine the datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Combine title and text for better predictions
df['content'] = df['title'] + ' ' + df['text']

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total samples: {len(df)}")
print(f"Fake news samples: {len(fake_df)}")
print(f"Real news samples: {len(true_df)}")

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df['content'],
    df['label'],
    test_size=0.2,
    random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Create TF-IDF vectorizer
# Using more features since we have a large dataset
print("\nVectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"Feature vector size: {X_train_vec.shape[1]}")

# Train a Logistic Regression model
print("\nTraining the model (this may take a minute)...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_vec, y_train)
print("Training complete!")

# Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Save the model and vectorizer
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fakenews_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n[OK] Model saved to model/fakenews_model.pkl")
print("[OK] Vectorizer saved to model/vectorizer.pkl")
print("\nYou can now run the Flask app with: python app.py")
