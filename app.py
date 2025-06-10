from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image

app = Flask(__name__)
import pickle

# Load trained model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


ALLOWED_EXTENSIONS = {'txt', 'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext in ['png', 'jpg', 'jpeg']:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)
    return ""

# 🔥 Your main route for user prediction
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        news = ""

        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                news = extract_text_from_file(file_path)

        if not news:
            news = request.form.get('news', '')

        if news.strip() != "":
            result = "PREDICTED: (You can plug in model here)"  # 🔁 Replace with your model.predict
        
        if news.strip() != "":
            news_vectorized = vectorizer.transform([news])
            result = model.predict(news_vectorized)[0]  # 🔥 Predict


    return render_template("index.html", result=result)

# 🔐 Admin panel for uploading new training data
@app.route('/admin', methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        true_file = request.files['true_data']
        fake_file = request.files['fake_data']
        if true_file and fake_file:
            true_path = os.path.join('dataset', 'True.csv')
            fake_path = os.path.join('dataset', 'Fake.csv')
            true_file.save(true_path)
            fake_file.save(fake_path)

            os.system('python train_model.py')

            return "✅ Model retrained successfully!"

    return render_template('admin.html')

# ✅ Run the app
if __name__ == "__main__":
    app.run(debug=True)
