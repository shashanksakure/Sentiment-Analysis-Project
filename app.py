from flask import Flask, request, render_template_string
import pickle
import os

app = Flask(__name__)

# ============================================================
# LOAD MODEL AND TF-IDF VECTORIZER
# ============================================================

MODEL_PATH = "model .pkl"
VECTORIZER_PATH = "vectorizer.pkl"

try:
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("Model and vectorizer loaded successfully!")

except Exception as e:
    print("Error loading model/vectorizer:", e)
    model = None
    vectorizer = None


# ============================================================
# HTML + CSS
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>SentimentAI | Sentiment Analysis</title>

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Arial, Helvetica, sans-serif;
        }

        body {
            min-height: 100vh;
            background:
                radial-gradient(circle at top left, #4f46e5, transparent 35%),
                radial-gradient(circle at bottom right, #06b6d4, transparent 35%),
                linear-gradient(135deg, #0f172a, #1e293b);

            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 1000px;
        }

        /* HEADER */

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .logo {
            display: inline-block;
            background: rgba(255,255,255,0.12);
            padding: 10px 18px;
            border-radius: 30px;
            backdrop-filter: blur(10px);
            margin-bottom: 15px;
            font-size: 15px;
            letter-spacing: 1px;
        }

        .header h1 {
            font-size: 48px;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #a5b4fc, #67e8f9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: #cbd5e1;
            font-size: 17px;
        }

        /* MAIN CARD */

        .card {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            border-radius: 25px;
            padding: 35px;
            backdrop-filter: blur(20px);
            box-shadow: 0 25px 60px rgba(0,0,0,0.35);
        }

        label {
            display: block;
            font-size: 17px;
            font-weight: bold;
            margin-bottom: 12px;
        }

        textarea {
            width: 100%;
            height: 180px;
            resize: none;

            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 18px;

            padding: 20px;

            background: rgba(15,23,42,0.65);
            color: white;

            font-size: 17px;
            outline: none;

            transition: 0.3s;
        }

        textarea:focus {
            border-color: #67e8f9;
            box-shadow: 0 0 20px rgba(103,232,249,0.2);
        }

        textarea::placeholder {
            color: #94a3b8;
        }

        /* BUTTON */

        .button-container {
            margin-top: 20px;
        }

        button {
            width: 100%;
            padding: 16px;

            border: none;
            border-radius: 14px;

            background: linear-gradient(
                90deg,
                #6366f1,
                #06b6d4
            );

            color: white;

            font-size: 17px;
            font-weight: bold;

            cursor: pointer;

            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(6,182,212,0.35);
        }

        /* RESULT */

        .result {
            margin-top: 30px;
            padding: 25px;

            border-radius: 18px;

            background: rgba(15,23,42,0.65);

            border: 1px solid rgba(255,255,255,0.15);

            text-align: center;
        }

        .result-title {
            color: #94a3b8;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 10px;
        }

        .sentiment {
            font-size: 34px;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .confidence {
            color: #cbd5e1;
            font-size: 15px;
        }

        /* SENTIMENT COLORS */

        .positive {
            color: #4ade80;
        }

        .negative {
            color: #fb7185;
        }

        .neutral {
            color: #facc15;
        }

        /* INFO CARDS */

        .features {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;

            margin-top: 25px;
        }

        .feature {
            padding: 20px;

            border-radius: 16px;

            background: rgba(255,255,255,0.06);

            border: 1px solid rgba(255,255,255,0.10);

            text-align: center;
        }

        .feature-icon {
            font-size: 30px;
            margin-bottom: 10px;
        }

        .feature h3 {
            font-size: 15px;
            margin-bottom: 5px;
        }

        .feature p {
            color: #94a3b8;
            font-size: 13px;
        }

        /* FOOTER */

        footer {
            text-align: center;
            margin-top: 25px;
            color: #94a3b8;
            font-size: 13px;
        }

        /* MOBILE */

        @media(max-width: 700px) {

            .header h1 {
                font-size: 36px;
            }

            .card {
                padding: 22px;
            }

            .features {
                grid-template-columns: 1fr;
            }

            body {
                padding: 15px;
            }

        }

    </style>

</head>

<body>

<div class="container">

    <!-- HEADER -->

    <div class="header">

        <div class="logo">
            🤖 AI POWERED NLP
        </div>

        <h1>SentimentAI</h1>

        <p>
            Analyze the emotional tone of your text using
            Machine Learning & NLP
        </p>

    </div>


    <!-- MAIN CARD -->

    <div class="card">

        <form method="POST">

            <label>
                💬 Enter your text
            </label>

            <textarea
                name="text"
                placeholder="Example: I really enjoyed this product. The quality is amazing!"
                required
            >{{ text }}</textarea>


            <div class="button-container">

                <button type="submit">
                    ✨ Analyze Sentiment
                </button>

            </div>

        </form>


        {% if prediction %}

        <div class="result">

            <div class="result-title">
                Analysis Result
            </div>

            <div class="sentiment {{ sentiment_class }}">

                {{ prediction }}

            </div>

            {% if confidence %}

            <div class="confidence">

                Model Confidence:
                <strong>{{ confidence }}%</strong>

            </div>

            {% endif %}

        </div>

        {% endif %}


        <!-- FEATURES -->

        <div class="features">

            <div class="feature">

                <div class="feature-icon">
                    🧠
                </div>

                <h3>Machine Learning</h3>

                <p>
                    Multinomial Naive Bayes model
                </p>

            </div>


            <div class="feature">

                <div class="feature-icon">
                    📊
                </div>

                <h3>TF-IDF</h3>

                <p>
                    Text feature extraction
                </p>

            </div>


            <div class="feature">

                <div class="feature-icon">
                    ⚡
                </div>

                <h3>Fast Prediction</h3>

                <p>
                    Real-time sentiment analysis
                </p>

            </div>

        </div>

    </div>


    <footer>

        SentimentAI • Flask • NLP • Machine Learning

    </footer>

</div>

</body>

</html>
"""


# ============================================================
# FLASK ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    sentiment_class = ""
    text = ""

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text and model is not None and vectorizer is not None:

            try:

                # Convert text into TF-IDF features
                text_vector = vectorizer.transform([text])

                # Predict sentiment
                result = model.predict(text_vector)[0]

                # Convert prediction to string
                prediction = str(result)

                # Get probability/confidence
                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(text_vector)

                    confidence = round(
                        float(max(probabilities[0])) * 100,
                        2
                    )

                # ------------------------------------------------
                # SENTIMENT CLASS
                # ------------------------------------------------

                sentiment_lower = prediction.lower()

                if "positive" in sentiment_lower:

                    sentiment_class = "positive"

                elif "negative" in sentiment_lower:

                    sentiment_class = "negative"

                else:

                    sentiment_class = "neutral"

            except Exception as e:

                prediction = "Prediction Error"
                sentiment_class = "negative"
                confidence = None

                print("Prediction error:", e)

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        sentiment_class=sentiment_class,
        text=text
    )


# ============================================================
# HEALTH CHECK
# Useful for AWS deployment
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
