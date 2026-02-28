import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# Add labels
fake['label'] = 0
true['label'] = 1

# Combine datasets
data = pd.concat([fake, true])
data = data.sample(frac=1).reset_index(drop=True)


# Keep only required columns
data = data[['text', 'label']]


print("✅ Dataset combined successfully!")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("🎯 Model Accuracy:", accuracy)
import pickle

# Save model
with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved!")