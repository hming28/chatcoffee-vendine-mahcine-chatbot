# ☕ ChatCoffee: Malaysian Unmanned Smart Kiosk
### TAR UMT — BMDS2123 Natural Language Processing (Topic 4 Assignment)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An autonomous, multi-turn conversational AI kiosk engineered specifically for authentic Malaysian multi-lingual environments (Kopitiam, food courts, mamak stalls). Powered by **Supervised Machine Learning Intent Classification**, **Dual-Script Named Entity Recognition (NER)**, **Deterministic Dialogue State Tracking (DST)**, and a **Nutritional RAG Knowledge Base**.

---

## 🚀 Key Highlights & Architectural Features

1. **Dual-Script Orthographic Normalization & Slang Engine**:
   - Integrated 341-rule normalization dictionary (`shortform-dict.json`) handling colloquial contractions (`nk` $\rightarrow$ `nak`, `xnak` $\rightarrow$ `tak nak`, `brp` $\rightarrow$ `berapa`).
   - CJK character preservation preventing substring collisions across English, Bahasa Melayu, and Chinese dialects (Cantonese, Hokkien).

2. **Supervised ML Intent Classifier (Zero Rule-Based Classification)**:
   - Evaluated across 4 algorithms (MultinomialNB, Logistic Regression, LinearSVC, Random Forest).
   - Champion Model: **Logistic Regression with TF-IDF N-Gram (1-2) Vectorization** trained on **850 curated gold instances (`chatcoffee_gold_850.csv`)**.
   - Outputs live Softmax probability distributions across 4 core intents:
     - `order_drink`
     - `clarify_order`
     - `modify_order`
     - `health_query`

3. **Dual-Script Named Entity Recognition (NER)**:
   - Extracts 5 beverage attributes: `base_drink`, `temperature`, `sweetness_level`, `milk_type`, and `serving_type`.
   - Native word-boundary and colloquial particle tolerance (`Peng`, `Ping`, `Siu Dai`, `Gah Dai`, `Kosong`, `Kau`, `Po`, `Tapau`).

4. **Deterministic Dialogue State Tracker (DST)**:
   - Proactive slot gathering for missing attributes (e.g. asking temperature before proceeding).
   - Dynamic mid-order human corrections (Task 7 benchmark) with dynamic price re-calculation.
   - Single-cup physical dispensing constraint handling.

5. **Nutritional RAG Knowledge Base**:
   - Grounded on 378 authentic beverage records (`unified_coffee_kb.csv`) providing laboratory-verified calories, sugar content, caffeine levels, and live RM checkout pricing.

---

## 📂 Project Structure

```text
├── index.html                                     # Interactive Smart Kiosk Web UI (Live Voice & ML)
├── app.py                                         # FastAPI server hosting REST API & Kiosk UI
├── ChatCoffee_NLP_Model_Training_and_Evaluation.ipynb # Complete Jupyter Notebook (EDA, ML & Evaluation)
├── chatcoffee_gold_850.csv                        # 850-instance Gold NLU Benchmark Dataset
├── unified_coffee_kb.csv                          # 378-drink Grounded Knowledge Base
├── requirements.txt                               # Production Python dependencies
├── Dockerfile                                     # 1-Click Container Deployment
├── Procfile                                       # Render / PaaS deployment entrypoint
├── models/
│   ├── chatcoffee_intent_classifier.joblib        # Serialized Scikit-Learn champion model
│   └── chatcoffee_model_web.json                  # Mathematical vector-matrix weights for Client Web
└── NLP_Datasets/
    ├── shortform-dict.json                        # 341 Malaysian slang normalization rules
    ├── kopitiam.csv                               # Kopitiam beverage domain ontology
    └── starbucks_drinkMenu_expanded.csv           # Nutritional baseline dataset
```

---

## 🛠️ Local Development & Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/hming28/chatcoffee-vending-mahcine-chatbot.git
cd chatcoffee-vending-mahcine-chatbot
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```
Visit `http://localhost:8000` in your browser to interact with the live Kiosk interface!

---

## 🌐 API Endpoints & Live Deployment

* **Live Kiosk URL**: [https://chatcoffee.onrender.com/](https://chatcoffee.onrender.com/)

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the interactive Smart Kiosk Web UI (`index.html`) |
| `GET` | `/api/health` | System health check, active model, and dataset statistics |
| `POST` | `/api/predict_intent` | Raw ML classification with Softmax class probabilities |
| `POST` | `/api/chat` | End-to-end multi-turn conversational dialogue turn |
| `POST` | `/api/tts` | High-fidelity Malaysian neural speech synthesis (1.25x speed) |
| `POST` | `/api/transcribe` | Audio file transcription endpoint |
