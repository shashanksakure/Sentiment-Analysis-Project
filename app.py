import os
import pickle
from flask import Flask, request, render_template_string

app = Flask(__name__)

# --- Load Pickled Files ---
# Replace filenames if yours are named differently (e.g., 'model.pkl', 'tfidf.pkl')
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, "rb") as f_model:
        model = pickle.load(f_model)
    with open(VECTORIZER_PATH, "rb") as f_vec:
        vectorizer = pickle.load(f_vec)
else:
    print("Warning: .pkl files not found! Ensure model.pkl and vectorizer.pkl are in the project folder.")

# --- Embedded Responsive & Attractive Layout ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.12);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-main);
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 16px;
            color: var(--text-main);
            font-size: 1rem;
            resize: none;
            outline: none;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        textarea:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.3);
        }

        button {
            background: linear-gradient(90deg, #7c3aed, #2563eb);
            color: #ffffff;
            font-weight: 600;
            font-size: 1rem;
            border: none;
            padding: 14px 24px;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s ease, opacity 0.2s ease;
        }

        button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 14px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
        }

        .positive {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.4);
            color: #4ade80;
        }

        .negative {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #f87171;
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 6px;
            opacity: 0.8;
        }

        .result-sentiment {
            font-size: 1.6rem;
            font-weight: 700;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Sentiment Analysis AI</h1>
            <p>Type or paste your text below to analyze its emotion.</p>
        </div>

        <form method="POST">
            <textarea name="text" placeholder="Enter your review or sentence here..." required>{{ text }}</textarea>
            <button type="submit">Analyze Sentiment</button>
        </form>

        {% if sentiment %}
        <div class="result-box {{ sentiment.lower() }}">
            <div class="result-title">Predicted Sentiment</div>
            <div class="result-sentiment">{{ sentiment }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    text_input = ""

    if request.method == "POST":
        text_input = request.form.get("text", "")
        if text_input and model and vectorizer:
            # Vectorize input text & Predict
            transformed_text = vectorizer.transform([text_input])
            prediction = model.predict(transformed_text)[0]
            
            # Map output string/integer label
            if str(prediction).lower() in ["1", "positive", "pos"]:
                sentiment = "Positive"
            else:
                sentiment = "Negative"
        elif not model or not vectorizer:
            sentiment = "Error: .pkl files missing!"

    return render_template_string(HTML_TEMPLATE, sentiment=sentiment, text=text_input)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
