from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news_text = request.form["news"]
        vector = vectorizer.transform([news_text])
        prediction = model.predict(vector)[0]
        result = f"📰 This news is: {prediction}"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
