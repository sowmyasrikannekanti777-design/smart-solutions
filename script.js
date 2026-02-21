/**
 * LearnSphere – script.js
 * Handles: Dark Mode, Navbar, Typing animation,
 *          AOS, Demo AI mock, Chatbot, Back-to-top
 */

/* ── AOS Init ── */
AOS.init({
  duration: 650,
  once: true,
  offset: 60,
});

/* ═══════════════════════════════════════
   1. DARK / LIGHT THEME TOGGLE
═══════════════════════════════════════ */
const themeToggle = document.getElementById('themeToggle');
const themeIcon   = document.getElementById('themeIcon');
const html        = document.documentElement;

const savedTheme = localStorage.getItem('ls-theme') || 'light';
html.setAttribute('data-theme', savedTheme);
updateThemeIcon(savedTheme);

themeToggle.addEventListener('click', () => {
  const current = html.getAttribute('data-theme');
  const next    = current === 'dark' ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('ls-theme', next);
  updateThemeIcon(next);
});

function updateThemeIcon(theme) {
  themeIcon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
}

/* ═══════════════════════════════════════
   2. NAVBAR – scroll + hamburger
═══════════════════════════════════════ */
const navbar    = document.getElementById('navbar');
const hamburger = document.getElementById('hamburger');
const navLinks  = document.getElementById('navLinks');

window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
  backToTop.classList.toggle('visible', window.scrollY > 400);
}, { passive: true });

hamburger.addEventListener('click', () => {
  navLinks.classList.toggle('open');
  const isOpen = navLinks.classList.contains('open');
  hamburger.setAttribute('aria-expanded', isOpen);
});

// Close menu on link click
navLinks.querySelectorAll('a').forEach(a => {
  a.addEventListener('click', () => navLinks.classList.remove('open'));
});

/* ═══════════════════════════════════════
   3. TYPING ANIMATION
═══════════════════════════════════════ */
const phrases = [
  'personalized AI explanations',
  'interactive code examples',
  'immersive audio lessons',
  'stunning visual diagrams',
  'adaptive learning paths',
];

let phraseIndex = 0;
let charIndex   = 0;
let deleting    = false;
const typingEl  = document.getElementById('typingTarget');

function type() {
  const currentPhrase = phrases[phraseIndex];
  if (!deleting) {
    typingEl.textContent = currentPhrase.slice(0, charIndex + 1);
    charIndex++;
    if (charIndex === currentPhrase.length) {
      deleting = true;
      setTimeout(type, 1800);
      return;
    }
  } else {
    typingEl.textContent = currentPhrase.slice(0, charIndex - 1);
    charIndex--;
    if (charIndex === 0) {
      deleting     = false;
      phraseIndex  = (phraseIndex + 1) % phrases.length;
    }
  }
  setTimeout(type, deleting ? 55 : 80);
}

type();

/* ═══════════════════════════════════════
   4. BACK TO TOP
═══════════════════════════════════════ */
const backToTop = document.getElementById('backToTop');
backToTop.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ═══════════════════════════════════════
   5. DEMO – MOCK AI RESPONSES
═══════════════════════════════════════ */
const generateBtn    = document.getElementById('generateBtn');
const topicInput     = document.getElementById('topicInput');
const demoPlaceholder = document.getElementById('demoPlaceholder');
const demoResponse   = document.getElementById('demoResponse');
const responseContent = document.getElementById('responseContent');
const responseMeta   = document.getElementById('responseMeta');
const copyBtn        = document.getElementById('copyBtn');

// Topic pill quick-fill
document.querySelectorAll('.topic-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('.topic-pill').forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    topicInput.value = pill.dataset.topic;
    topicInput.focus();
  });
});

// Mock knowledge base
const mockResponses = {
  explanation: {
    default: (topic) => `**${topic}**\n\nAt its core, ${topic} is a fundamental technique in machine learning that enables computers to learn patterns from data without being explicitly programmed.\n\n**How it works:**\n1. The model receives labeled training data\n2. It identifies statistical patterns and correlations\n3. These patterns are encoded into the model's parameters\n4. The trained model generalises to unseen data\n\n**Real-world analogy:**\nThink of it like a student studying many examples before an exam. The more quality examples they study, the better they generalise to new, unseen questions.\n\n**Key components:**\n• Training data — the examples the model learns from\n• Loss function — measures prediction errors\n• Optimiser — adjusts model weights to reduce loss\n• Evaluation metrics — assess performance on unseen data\n\n💡 **LearnSphere Tip:** Start with a simple linear model before jumping to complex architectures. Understanding the basics deeply is far more valuable than memorising advanced techniques.`,

    'Neural Networks': `**Neural Networks**\n\nNeural Networks are computational systems loosely inspired by the biological neural networks in the human brain. They consist of layers of interconnected nodes (neurons) that transform input data into meaningful outputs.\n\n**Architecture:**\n• **Input Layer** — receives raw data (pixels, features, tokens)\n• **Hidden Layers** — extract increasingly abstract representations\n• **Output Layer** — produces predictions or classifications\n\n**How learning works:**\n1. Data flows forward through the network (forward pass)\n2. Error is measured using a loss function\n3. Gradients flow backwards (backpropagation)\n4. Weights are updated using gradient descent\n\n**Activation Functions:**\nReLU, Sigmoid, Tanh, GELU — these introduce non-linearity, enabling the network to learn complex patterns.\n\n💡 **LearnSphere Tip:** \`torch.nn.Module\` is your best friend in PyTorch. Every neural network you build will inherit from it.`,

    'Gradient Descent': `**Gradient Descent**\n\nGradient Descent is the optimisation algorithm that powers almost all of modern machine learning. It finds the minimum of a loss function by iteratively moving in the direction of steepest descent.\n\n**The Core Idea:**\nImagine you're blindfolded on a hilly landscape and want to reach the valley. You feel the slope under your feet and take a step downhill. You repeat this until you reach flat ground. That's gradient descent.\n\n**Update Rule:**\n\`w = w - learning_rate × ∂Loss/∂w\`\n\n**Variants:**\n• **Batch GD** — uses entire dataset, stable but slow\n• **Stochastic GD (SGD)** — uses one sample, fast but noisy\n• **Mini-batch GD** — uses small batches, best of both worlds\n• **Adam** — adaptive learning rates, most widely used today\n\n**Learning Rate:**\nToo high → loss diverges. Too low → training is painfully slow. Use learning rate schedulers for best results.\n\n💡 **LearnSphere Tip:** Always visualise your loss curve. A decreasing then plateauing curve is healthy. Spikes may indicate a learning rate that's too high.`,

    'Random Forest': `**Random Forest**\n\nRandom Forest is an ensemble learning method that builds multiple decision trees and merges their predictions for more accurate, robust results. It's one of the most reliable algorithms in classical machine learning.\n\n**Core Mechanism — Bagging:**\n1. Sample N subsets of training data with replacement (bootstrapping)\n2. Train one decision tree on each subset\n3. Each tree also uses a random subset of features at each split\n4. Aggregrate predictions: voting (classification) or averaging (regression)\n\n**Strengths:**\n✅ Resistant to overfitting\n✅ Handles missing values well\n✅ Provides feature importance scores\n✅ Works well on tabular data out of the box\n\n**Key Hyperparameters:**\n• \`n_estimators\` — number of trees (more = better, diminishing returns)\n• \`max_depth\` — controls tree complexity\n• \`max_features\` — features per split (√n for classification)\n\n💡 **LearnSphere Tip:** Random Forest is an excellent first model to try on any tabular dataset. It's interpretable, fast to train, and rarely fails catastrophically.`,
  },
  code: {
    default: (topic) =>
`\`\`\`python
# LearnSphere Code Example: ${topic}
# Generated by Gemini AI

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ── Step 1: Generate sample dataset ──────────────────────────
np.random.seed(42)
X = np.random.randn(500, 4)          # 500 samples, 4 features
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary target

# ── Step 2: Split into train and test sets ────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Step 3: Normalise features (important for many algorithms) ─
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # Fit on train only!
X_test  = scaler.transform(X_test)       # Apply same transform

# ── Step 4: Implement ${topic} ──────────────────────
from sklearn.ensemble import RandomForestClassifier  # Replace with desired model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Step 5: Evaluate the model ───────────────────────────────
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# ── Step 6: Feature importance (where applicable) ────────────
importances = model.feature_importances_
print("Feature importances:", np.round(importances, 3))
\`\`\`

Run this in Google Colab or Jupyter Notebook. All dependencies are available via \`pip install scikit-learn numpy\`.`,
  },
  quiz: {
    default: (topic) =>
`**Quiz: Test Your ${topic} Knowledge**\n\n**Q1.** What is the primary goal of ${topic}?\n   a) Memorise training data\n   b) Learn generalisable patterns\n   c) Minimise dataset size\n   d) Maximise model complexity\n\n**Q2.** Which of the following is NOT a type of machine learning?\n   a) Supervised Learning\n   b) Reinforced Learning\n   c) Deterministic Learning ✓ (correct answer)\n   d) Unsupervised Learning\n\n**Q3.** What does overfitting mean?\n   a) Model performs equally on train and test data\n   b) Model performs well on training but poorly on new data ✓\n   c) Model has too few parameters\n   d) Training loss is too high\n\n**Q4.** Which metric is most appropriate for imbalanced classification?\n   a) Accuracy\n   b) F1-Score ✓\n   c) Mean Squared Error\n   d) R-Squared\n\n📊 Your score will appear after you attempt all questions!\n💡 **LearnSphere adapts** the next set of questions based on which answers you get wrong.`,
  },
};

function getResponse(topic, mode) {
  const key = topic.trim();
  if (mode === 'code') {
    return mockResponses.code[key] || mockResponses.code.default(topic);
  } else if (mode === 'quiz') {
    return mockResponses.quiz[key] || mockResponses.quiz.default(topic);
  } else {
    return mockResponses.explanation[key] || mockResponses.explanation.default(topic);
  }
}

function showLoader() {
  demoPlaceholder.style.display = 'none';
  demoResponse.classList.add('hidden');
  demoResponse.style.display = 'none';

  const loader = document.createElement('div');
  loader.id = 'demoLoader';
  loader.className = 'loader';
  loader.innerHTML = `
    <span>Gemini AI is generating your response</span>
    <div class="loader-dots">
      <span></span><span></span><span></span>
    </div>`;
  document.getElementById('demoOutputArea').appendChild(loader);
}

function hideLoader() {
  const loader = document.getElementById('demoLoader');
  if (loader) loader.remove();
}

function animateText(el, text) {
  el.textContent = '';
  let i = 0;
  const interval = setInterval(() => {
    el.textContent += text.charAt(i);
    i++;
    if (i >= text.length) clearInterval(interval);
  }, 10);
}

function handleGenerate() {
  const topic = topicInput.value.trim();
  if (!topic) {
    topicInput.focus();
    topicInput.style.borderColor = '#FF6B9D';
    setTimeout(() => (topicInput.style.borderColor = ''), 1500);
    return;
  }

  const mode = document.querySelector('input[name="demoMode"]:checked').value;

  showLoader();
  generateBtn.disabled = true;

  // Simulate AI delay
  const delay = 900 + Math.random() * 600;
  setTimeout(() => {
    hideLoader();
    const text = getResponse(topic, mode);

    demoPlaceholder.style.display = 'none';
    demoResponse.style.display = 'block';
    demoResponse.classList.remove('hidden');
    animateText(responseContent, text);

    const now = new Date();
    responseMeta.textContent = `Generated for "${topic}" · ${mode} mode · ${now.toLocaleTimeString()}`;
    generateBtn.disabled = false;
  }, delay);
}

generateBtn.addEventListener('click', handleGenerate);
topicInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') handleGenerate();
});

// Copy button
copyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(responseContent.textContent).then(() => {
    copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
    setTimeout(() => (copyBtn.innerHTML = '<i class="fa-solid fa-copy"></i>'), 2000);
  });
});

// Audio / Code buttons inside response
document.getElementById('audioBtn').addEventListener('click', () => {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(responseContent.textContent);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  } else {
    alert('Audio synthesis not supported in your browser.');
  }
});

document.getElementById('codeBtn').addEventListener('click', () => {
  const topic = topicInput.value.trim() || 'Machine Learning';
  document.querySelector('input[name="demoMode"][value="code"]').checked = true;
  topicInput.value = topic;
  handleGenerate();
});

/* ═══════════════════════════════════════
   6. CHATBOT MOCK
═══════════════════════════════════════ */
const chatMessages = document.getElementById('chatMessages');
const chatInput    = document.getElementById('chatInput');
const chatSend     = document.getElementById('chatSend');

const chatKnowledge = {
  'hello': 'Hello! 👋 I\'m Sphere, your AI tutor. Which ML concept would you like to explore?',
  'hi': 'Hi there! Ready to learn some Machine Learning today? Ask me anything!',
  'what is ml': 'Machine Learning is a branch of AI where models learn patterns from data rather than being explicitly programmed. It powers recommendations, image recognition, speech-to-text, and much more!',
  'overfitting': 'Overfitting occurs when a model memorises training data so well that it fails to generalise to new, unseen data. Fix it with: regularisation (L1/L2), dropout, cross-validation, or more training data.',
  'neural network': 'Neural networks are systems of interconnected nodes inspired by the brain. They learn by adjusting weights via backpropagation. Deep learning = neural networks with many hidden layers!',
  'python': 'Python is the language of choice for ML! Key libraries: NumPy (math), Pandas (data), Scikit-learn (classical ML), PyTorch & TensorFlow (deep learning), and Matplotlib/Seaborn (viz).',
  'flask': 'Flask is a lightweight Python web framework. In LearnSphere, it powers the backend API that bridges the frontend with Gemini AI. It\'s perfect for rapid prototyping.',
  'gemini': 'Google Gemini is the generative AI engine powering LearnSphere. It generates explanations, code, and quiz questions on-the-fly, personalised to each learner.',
  'default': [
    'Great question! Let me think about that... 🤔 This topic is covered in our full platform. Would you like a text explanation or code example?',
    'Interesting topic! In ML, the best way to understand it is through hands-on experimentation. Want me to show you a code snippet?',
    'That\'s a concept worth exploring deeply. Our Gemini AI engine can generate a tailored explanation for you — try using the Demo panel on the left!',
    'I\'d love to answer that in detail. The full LearnSphere platform adapts explanations to your level. What\'s your current ML experience?',
  ],
};

function addChatMessage(text, sender) {
  const msg  = document.createElement('div');
  msg.className = `chat-msg ${sender}`;
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.textContent = text;
  msg.appendChild(bubble);
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
  const indicator = document.createElement('div');
  indicator.id = 'typingIndicator';
  indicator.className = 'chat-msg bot';
  indicator.innerHTML = `
    <div class="chat-bubble">
      <div class="loader-dots">
        <span></span><span></span><span></span>
      </div>
    </div>`;
  chatMessages.appendChild(indicator);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function getBotResponse(input) {
  const lower = input.toLowerCase().trim();
  for (const [key, val] of Object.entries(chatKnowledge)) {
    if (lower.includes(key)) return val;
  }
  const defaults = chatKnowledge.default;
  return defaults[Math.floor(Math.random() * defaults.length)];
}

function sendChat() {
  const text = chatInput.value.trim();
  if (!text) return;

  addChatMessage(text, 'user');
  chatInput.value = '';

  showTypingIndicator();

  setTimeout(() => {
    removeTypingIndicator();
    addChatMessage(getBotResponse(text), 'bot');
  }, 800 + Math.random() * 500);
}

chatSend.addEventListener('click', sendChat);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendChat();
});

/* ═══════════════════════════════════════
   7. ACTIVE NAV LINK (IntersectionObserver)
═══════════════════════════════════════ */
const sections = document.querySelectorAll('section[id]');
const navAs    = document.querySelectorAll('.nav-links a');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        navAs.forEach((a) => {
          a.classList.toggle('active', a.getAttribute('href') === `#${id}`);
        });
      }
    });
  },
  { rootMargin: '-30% 0px -60% 0px' }
);

sections.forEach((s) => observer.observe(s));

// Inject active nav link CSS
const style = document.createElement('style');
style.textContent = `.nav-links a.active { color: var(--primary) !important; }`;
document.head.appendChild(style);
