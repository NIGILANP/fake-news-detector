import pickle

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Test input
news = input("Enter news: ")

# Convert text
vector = vectorizer.transform([news])

# Predict
prediction = model.predict(vector)
prob = model.predict_proba(vector)[0]
confidence = max(prob) * 100

# Output
if prediction[0] == 1:
    print("✅ Real News")
    print(f"Confidence: {confidence:.2f}%")
else:
    print("❌ Fake News")
    print(f"Confidence: {confidence:.2f}%")