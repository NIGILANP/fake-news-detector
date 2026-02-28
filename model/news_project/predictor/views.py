from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from .models import NewsHistory
import pickle
import requests
import os
from django.conf import settings

# 📁 Load model
BASE_DIR = settings.BASE_DIR
ROOT_DIR = os.path.dirname(BASE_DIR)

model_path = os.path.join(ROOT_DIR, "fake_news_model.pkl")
vectorizer_path = os.path.join(ROOT_DIR, "vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

history = []

def summarize(text):
    return " ".join(text.split()[:20])


# 🔐 SIGNUP VIEW
def signup(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)

        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("home")
        else:
            print(form.errors)  # 🔴 DEBUG

    else:
        form = UserCreationForm()

    return render(request, "signup.html", {"form": form})


# 🧠 HOME VIEW (PROTECTED)
@login_required
def home(request):
    prediction = None
    summary = None

    if request.method == "POST":
        text = request.POST.get("news")

        vect = vectorizer.transform([text])
        result = model.predict(vect)[0]

        prob = model.predict_proba(vect)[0]
        confidence = max(prob) * 100

        summary = summarize(text)

        if result == 1:
            prediction = "Real News"
        else:
            prediction = "Fake News"

        # 💾 SAVE TO DATABASE
        NewsHistory.objects.create(
            user=request.user,
            text=text,
            prediction=prediction,
            confidence=confidence
        )

    return render(request, "index.html", {
        "prediction": prediction,
        "summary": summary,
    })

@login_required
def dashboard(request):
    data = NewsHistory.objects.filter(user=request.user).order_by('-created_at')

    total = data.count()
    real = data.filter(prediction="Real News").count()
    fake = data.filter(prediction="Fake News").count()

    return render(request, "dashboard.html", {
        "data": data,
        "total": total,
        "real": real,
        "fake": fake
    })    
    


@login_required
def latest_news(request):
    url = "https://newsapi.org/v2/top-headlines"

    params = {
        "country": "us",
        "apiKey": "ccf68b193d5849a7821ee29128ef24f2"
    }

    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get("articles", [])

    processed_articles = []

    for article in articles:
        title = article.get("title", "")

        if title:
            vect = vectorizer.transform([title])
            result = model.predict(vect)[0]
            prob = model.predict_proba(vect)[0]
            confidence = max(prob) * 100

            if result == 1:
                prediction = "Real"
                badge = "real"
            else:
                prediction = "Fake"
                badge = "fake"

            processed_articles.append({
                "title": title,
                "description": article.get("description"),
                "url": article.get("url"),
                "prediction": prediction,
                "confidence": f"{confidence:.2f}",
                "badge": badge
            })

    return render(request, "news.html", {"articles": processed_articles})