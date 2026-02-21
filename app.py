"""
LearnSphere – app.py
Flask backend for the AI-powered ML Learning System.
Integrates with Google Gemini AI for content generation.
"""

from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os

# ── App Initialisation ────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'learnsphere-dev-key')

# ── Gemini AI Setup ───────────────────────────────────────────
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY_HERE')
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# ── System Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are LearnSphere, an expert AI tutor specialising in Machine Learning education.
Your explanations are:
- Clear, jargon-free, and adapted to the user's level
- Supported by real-world analogies
- Structured with headings, bullet points, and examples
- Encouraging and patient
Always end with a practical tip labelled "💡 LearnSphere Tip:".
"""

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main LearnSphere website."""
    return render_template('index.html')


@app.route('/api/explain', methods=['POST'])
def explain():
    """
    Generate a personalised ML concept explanation.

    Request JSON:
        topic (str): The ML concept to explain.
        level (str): 'beginner', 'intermediate', or 'advanced'. Default: 'beginner'.

    Response JSON:
        explanation (str): AI-generated explanation.
        topic (str): Echo of the requested topic.
    """
    data  = request.get_json(silent=True) or {}
    topic = data.get('topic', '').strip()
    level = data.get('level', 'beginner')

    if not topic:
        return jsonify({'error': 'Topic is required.'}), 400

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Explain '{topic}' to a {level}-level learner. "
        f"Include: definition, how it works, key components, a real-world analogy, and a practical tip."
    )

    try:
        response = model.generate_content(prompt)
        return jsonify({
            'topic': topic,
            'level': level,
            'explanation': response.text,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/code', methods=['POST'])
def generate_code():
    """
    Generate a Python code example for an ML concept.

    Request JSON:
        topic (str): The ML concept.
        library (str): Preferred library ('sklearn', 'pytorch', 'tensorflow'). Default: 'sklearn'.

    Response JSON:
        code (str): AI-generated Python code with comments.
    """
    data    = request.get_json(silent=True) or {}
    topic   = data.get('topic', '').strip()
    library = data.get('library', 'sklearn')

    if not topic:
        return jsonify({'error': 'Topic is required.'}), 400

    prompt = (
        f"Write a clean, well-commented Python code example demonstrating '{topic}' "
        f"using {library}. Include: imports, data preparation, model training, evaluation, "
        f"and a brief explanation above each block as comments."
    )

    try:
        response = model.generate_content(prompt)
        return jsonify({
            'topic': topic,
            'library': library,
            'code': response.text,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/quiz', methods=['POST'])
def generate_quiz():
    """
    Generate a short quiz for an ML topic.

    Request JSON:
        topic (str): The ML concept.
        num_questions (int): Number of questions (1-10). Default: 4.

    Response JSON:
        quiz (str): AI-generated quiz with answers.
    """
    data          = request.get_json(silent=True) or {}
    topic         = data.get('topic', '').strip()
    num_questions = min(max(int(data.get('num_questions', 4)), 1), 10)

    if not topic:
        return jsonify({'error': 'Topic is required.'}), 400

    prompt = (
        f"Create a {num_questions}-question multiple-choice quiz about '{topic}' in Machine Learning. "
        f"Each question should have 4 options (a, b, c, d). Mark the correct answer with ✓. "
        f"Include a brief explanation for each correct answer."
    )

    try:
        response = model.generate_content(prompt)
        return jsonify({
            'topic': topic,
            'num_questions': num_questions,
            'quiz': response.text,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Multi-turn chatbot endpoint.

    Request JSON:
        message (str): User's message.
        history (list): List of {"role": "user"|"model", "parts": [str]} dicts.

    Response JSON:
        reply (str): AI tutor's response.
    """
    data    = request.get_json(silent=True) or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])

    if not message:
        return jsonify({'error': 'Message is required.'}), 400

    chat_session = model.start_chat(history=history)

    try:
        response = chat_session.send_message(
            f"{SYSTEM_PROMPT}\n\nUser: {message}"
        )
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/topics', methods=['GET'])
def list_topics():
    """Return a curated list of ML topics available in LearnSphere."""
    topics = [
        {"category": "Foundations",       "items": ["Linear Regression", "Logistic Regression", "Feature Engineering", "Train/Test Split", "Cross-Validation"]},
        {"category": "Classical ML",      "items": ["Decision Trees", "Random Forest", "Support Vector Machines", "K-Means Clustering", "Naive Bayes"]},
        {"category": "Deep Learning",     "items": ["Neural Networks", "Backpropagation", "Gradient Descent", "CNNs", "RNNs", "LSTMs"]},
        {"category": "Modern AI",         "items": ["Transformers", "Attention Mechanism", "BERT", "GPT Architecture", "Diffusion Models"]},
        {"category": "ML Engineering",    "items": ["Pipeline Design", "Hyperparameter Tuning", "Model Deployment", "MLflow", "Docker for ML"]},
    ]
    return jsonify({'topics': topics, 'total': sum(len(t['items']) for t in topics)})


@app.route('/health')
def health():
    """Simple health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'LearnSphere API'})


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("🚀 LearnSphere server starting...")
    print("📍 Visit: http://localhost:5000")
    print("⚠️  Set GEMINI_API_KEY environment variable before using AI features.")
    app.run(debug=True, host='0.0.0.0', port=5000)
