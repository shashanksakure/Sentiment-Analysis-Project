from flask import Flask, request, render_template_string
import pickle
import os

app = Flask(__name__)

# Load the trained sentiment-analysis model and TF-IDF vectorizer.
# Keep these two .pkl files in the same folder as app.py.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model .pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SentimentAI | Sentiment Analysis</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            min-height: 100vh;
            font-family: Inter, Arial, sans-serif;
            background:
                radial-gradient(circle at 10% 10%, rgba(99,102,241,.25), transparent 28%),
                radial-gradient(circle at 90% 80%, rgba(14,165,233,.22), transparent 30%),
                linear-gradient(135deg, #0f172a, #111827 55%, #172554);
            color: #f8fafc;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 30px 16px;
        }

        .container { width: 100%; max-width: 1050px; }

        .hero {
            text-align: center;
            margin-bottom: 24px;
        }

        .badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,.09);
            border: 1px solid rgba(255,255,255,.15);
            color: #c7d2fe;
            font-size: 13px;
            margin-bottom: 14px;
        }

        h1 {
            font-size: clamp(34px, 6vw, 62px);
            line-height: 1.05;
            letter-spacing: -2px;
        }

        h1 span {
            background: linear-gradient(90deg, #818cf8, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            max-width: 680px;
            margin: 14px auto 0;
            color: #cbd5e1;
            line-height: 1.7;
            font-size: 16px;
        }

        .card {
            background: rgba(15, 23, 42, .78);
            border: 1px solid rgba(255,255,255,.12);
            backdrop-filter: blur(18px);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 25px 70px rgba(0,0,0,.35);
        }

        label {
            display: block;
            font-weight: 700;
            margin-bottom: 10px;
            color: #e2e8f0;
        }

        textarea {
            width: 100%;
            min-height: 190px;
            resize: vertical;
            border: 1px solid #334155;
            background: #0b1220;
            color: #f8fafc;
            border-radius: 16px;
            padding: 18px;
            font-size: 16px;
            outline: none;
            transition: .2s ease;
        }

        textarea:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129,140,248,.12);
        }

        textarea::placeholder { color: #64748b; }

        .actions {
            display: flex;
            gap: 12px;
            margin-top: 16px;
            flex-wrap: wrap;
        }

        button {
            border: none;
            border-radius: 13px;
            padding: 13px 22px;
            font-size: 15px;
            font-weight: 700;
            cursor: pointer;
            transition: transform .2s, opacity .2s;
        }

        button:hover { transform: translateY(-2px); }
        .primary {
            background: linear-gradient(90deg, #6366f1, #0ea5e9);
            color: white;
            flex: 1;
        }

        .secondary {
            background: #1e293b;
            color: #cbd5e1;
        }

        .result {
            margin-top: 25px;
            padding: 22px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.12);
            background: rgba(255,255,255,.05);
            text-align: center;
        }

        .result.positive {
            border-color: rgba(34,197,94,.4);
            background: rgba(34,197,94,.08);
        }

        .result.negative {
            border-color: rgba(239,68,68,.4);
            background: rgba(239,68,68,.08);
        }

        .emoji { font-size: 44px; margin-bottom: 8px; }
        .result h2 { font-size: 27px; margin-bottom: 7px; }
        .result p { color: #cbd5e1; }

        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-top: 20px;
        }

        .feature {
            padding: 18px;
            border-radius: 16px;
            background: rgba(255,255,255,.045);
            border: 1px solid rgba(255,255,255,.08);
        }

        .feature strong { display: block; margin-bottom: 6px; }
        .feature small { color: #94a3b8; line-height: 1.5; }

        footer {
            text-align: center;
            margin-top: 22px;
            color: #64748b;
            font-size: 13px;
        }

        @media (max-width: 700px) {
            .card { padding: 20px; }
            .features { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="hero">
        <div class="badge">🤖 AI • NLP • Machine Learning</div>
        <h1>Sentiment<span>AI</span></h1>
        <p class="subtitle">
            Analyze the emotional tone of customer reviews, feedback, comments and other text
            using a trained TF-IDF + Naive Bayes machine-learning model.
        </p>
    </div>

    <div class="card">
        <form method="POST">
            <label for="text">Enter your text</label>
            <textarea id="text" name="text"
                placeholder="Example: I absolutely loved this product. The quality is excellent!"
                required>{{ text }}</textarea>

            <div class="actions">
                <button class="primary" type="submit">✨ Analyze Sentiment</button>
                <button class="secondary" type="button" onclick="document.getElementById('text').value=''">
                    Clear
                </button>
            </div>
        </form>

        {% if result %}
        <div class="result {{ result_class }}">
            <div class="emoji">{{ emoji }}</div>
            <h2>{{ result }}</h2>
            <p>{{ message }}</p>
        </div>
        {% endif %}

        <div class="features">
            <div class="feature">
                <strong>🧠 ML Model</strong>
                <small>Multinomial Naive Bayes trained for binary sentiment classification.</small>
            </div>
            <div class="feature">
                <strong>🔤 NLP</strong>
                <small>TF-IDF converts the input text into numerical features for prediction.</small>
            </div>
            <div class="feature">
                <strong>⚡ Flask</strong>
                <small>Lightweight web application ready for deployment on Render.</small>
            </div>
        </div>
    </div>

    <footer>SentimentAI • Built with Python, Flask, Scikit-learn & NLP</footer>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    result_class = ""
    emoji = ""
    message = ""
    text = ""

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if text:
            # Transform text using the SAME TF-IDF vectorizer used during training.
            features = vectorizer.transform([text])

            prediction = model.predict(features)[0]
            prediction = str(prediction).lower()

            if prediction == "positive":
                result = "Positive Sentiment"
                result_class = "positive"
                emoji = "😊"
                message = "The model predicts that this text expresses a positive sentiment."
            elif prediction == "negative":
                result = "Negative Sentiment"
                result_class = "negative"
                emoji = "😞"
                message = "The model predicts that this text expresses a negative sentiment."
            else:
                result = f"Predicted Sentiment: {prediction.title()}"
                result_class = ""
                emoji = "🔎"
                message = "The model has returned a sentiment classification."

    return render_template_string(
        HTML,
        result=result,
        result_class=result_class,
        emoji=emoji,
        message=message,
        text=text
    )


if __name__ == "__main__":
    # Render provides the PORT environment variable.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
