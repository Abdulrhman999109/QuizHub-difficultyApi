import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.corpus import words
nltk.download('words')
english_vocab = set(w.lower() for w in words.words())

def contains_valid_words(text):
    words_list = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    valid_count = sum(1 for w in words_list if w in english_vocab)
    return valid_count / max(len(words_list), 1) >= 0.5

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

class DifficultyPredictor(nn.Module):
    def __init__(self):
        super(DifficultyPredictor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.regressor(pooled_output)

MODEL_PATH = "difficulty_pred.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DifficultyPredictor().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
model.eval()

question_detector = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

def is_question(text):
    original_text = text.strip()

    if len(original_text.split()) < 2:
        return False
    if re.fullmatch(r'\d+', original_text):
        return False
    if not re.search(r'[a-zA-Zأ-ي]', original_text):
        return False

    is_arabic = bool(re.search(r'[\u0600-\u06FF]', original_text))
    if is_arabic:
        return True  

    cleaned_text = re.sub(r'[؟?]+$', '', original_text).strip()

    question_words = [
        "what", "how", "why", "who", "when", "where",
        "is", "are", "do", "does", "can", "define", "explain", "describe"
    ]

    words = cleaned_text.lower().split()
    unique_words = set(words)
    if len(unique_words) == 1 or all(w in question_words for w in unique_words):
        return False

    if not contains_valid_words(cleaned_text):
        return False

    labels = ["question", "not a question"]
    result = question_detector(cleaned_text, candidate_labels=labels)
    top_label = result['labels'][0]
    top_score = result['scores'][0]

    starts_with_question_word = any(cleaned_text.lower().startswith(q) for q in question_words)

    if top_score >= 0.90:
        return top_label == "question"
    elif top_score >= 0.85:
        return top_label == "question" and starts_with_question_word
    else:
        return False

def preprocess_text(text):
    tokens = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return tokens["input_ids"].to(device), tokens["attention_mask"].to(device)

def predict_difficulty(question_text):
    question_text = question_text.lower()
    input_ids, attention_mask = preprocess_text(question_text)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        difficulty_score = output.item()

    difficulty_score = max(0, min(difficulty_score, 100))
    return round(difficulty_score, 2)

app = Flask(__name__)
CORS(app)

@app.route('/api/questions/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        question = data.get('text')

        if not question:
            return jsonify({'error': 'Question text cannot be empty'}), 400

        if not is_question(question):
            return jsonify({'error': 'Please enter a valid question.'}), 400

        score = predict_difficulty(question)
        return jsonify({'difficulty_score': score})

    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
