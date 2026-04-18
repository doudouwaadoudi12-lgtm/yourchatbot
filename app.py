from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# FAQ
with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

questions = list(faq.keys())
answers = list(faq.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""

    if request.method == "POST":
        user_input = request.form.get("question")

        user_vec = vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, X)

        index = similarity.argmax()
        result = answers[index]

    return render_template(
        "index.html",
        result=result,
        suggestions=questions   # 👈 نبعث الاقتراحات
    )

if __name__ == "__main__":
    app.run(debug=True)