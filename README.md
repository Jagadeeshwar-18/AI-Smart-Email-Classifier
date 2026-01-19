# AI-Smart-Email-Classifier

## 🚀 Project Title
**AI-Powered Smart Email Classifier for Enterprises**

---

## 📌 Problem Statement
Enterprises receive a massive volume of emails every day. Manual email triaging is:
- Time-consuming
- Error-prone
- Inefficient for prioritizing critical communications

This project addresses the need for **automated email classification and urgency detection** to improve enterprise productivity and response time.

---

## 🎯 Project Objectives
- Automatically classify emails into predefined categories
- Detect urgency levels (High / Medium / Low)
- Reduce manual email handling workload
- Improve enterprise response efficiency

---

## 🧠 Core Capabilities
- Natural Language Processing (NLP)
- Machine Learning & Transformer-based models
- Hybrid Rule-based + ML decision system
- Scalable, enterprise-ready architecture

---

## 🏗️ System Architecture
Raw Emails
↓
Preprocessing
↓
Email Categorization
↓
Urgency Detection
↓
Final Output
(Category + Urgency)

yaml
Copy code

---

## ✅ Implemented Milestones

### ✔ Milestone 1: Data Preparation & Preprocessing
- Email cleaning (HTML tags, URLs, signatures removal)
- Manual labeling for category and urgency
- Dataset merging and train/validation/test split

---

### ✔ Milestone 2: Email Categorization Engine

#### Baseline Models
- Logistic Regression (TF-IDF)
- Naive Bayes (TF-IDF)

**Baseline Performance**
- Accuracy: ~92%
- Macro F1-score: ~92%

#### Transformer-Based Model
- Fine-tuned **DistilBERT** for multi-class email categorization

**DistilBERT Performance**
- Accuracy: **94.17%**
- Macro F1-score: **94.53%**

---

### ✔ Milestone 3: Urgency Detection Module

A **hybrid urgency detection system** was implemented.

#### Rule-Based Detection
- Identifies explicit urgency keywords (e.g., *urgent*, *ASAP*, *system down*)

#### ML-Based Detection
- Logistic Regression with TF-IDF
- Predicts urgency levels: **High / Medium / Low**

**Urgency Model Performance**
- Accuracy: **95%**
- Weighted F1-score: **0.95**
- Macro F1-score: **0.91**

#### Hybrid Decision Logic
- Rule-based **HIGH urgency overrides** ML predictions
- ML handles ambiguous and nuanced cases

---

## 📊 Results Summary

| Task                 | Model               | Accuracy | Macro F1 |
|----------------------|---------------------|----------|----------|
| Email Categorization | Logistic Regression | ~92%     | ~92%     |
| Email Categorization | DistilBERT          | 94.17%   | 94.53%   |
| Urgency Detection    | Hybrid (Rules + ML) | 95%      | 0.91     |

---

## 📁 Project Structure
```Text
AI-Smart-Email-Classifier/
│
├── data/
│ ├── raw/
│ ├── interim/
│ ├── processed/
│ └── splits/
│
├── src/
│ ├── preprocessing/
│ │ ├── cleaner.py
│ │ ├── generate_cleaned_csv.py
│ │ ├── label_categories.py
│ │ ├── label_urgency.py
│ │ ├── merge_datasets.py
│ │ └── split_dataset.py
│ │
│ └── models/
│ ├── vectorizer.py
│ ├── label_mapping.py
│ ├── train_logistic_regression.py
│ ├── train_naive_bayes.py
│ ├── train_distilbert.py
│ ├── evaluate_distilbert.py
│ ├── train_urgency_model.py
│ ├── urgency_rules.py
│ └── hybrid_urgency.py
│
├── models/
│ ├── category/
│ └── urgency/
│
├── README.md
├── LICENSE
└── .gitignore

```

> **Note:** Large datasets and trained models are excluded from version control.

---

## ▶️ How to Run

### Preprocess Emails
```bash
python src/preprocessing/generate_cleaned_csv.py
```
### Split Dataset
```bash

python src/preprocessing/split_dataset.py
```
### Train Category Models
```bash

python src/models/train_logistic_regression.py
python src/models/train_distilbert.py
```
Train Urgency Model
```bash

python src/models/train_urgency_model.py
```
### 🧪 Hybrid Urgency Logic
- Rule-based HIGH urgency overrides ML prediction

- ML model handles remaining cases

- Ensures high precision for critical emails
 ---

### 🚧 Excluded from Version Control
- Raw datasets

- Processed datasets

- Trained models

- Virtual environments
---

### 🌐 Deployed Application URL
- 🔗 Deployed URL: (https://ai-smart-email-classifier-vjf5o9fzr4vt4j7mekchhr.streamlit.app/)
---
### 🚀 Future Enhancements
- FastAPI backend integration

- Streamlit dashboard

- Docker-based deployment

- Cloud hosting (AWS / Azure / GCP)
---
### 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

---
### 🎯 Project Status
✔ Data pipeline completed

✔ Email categorization completed

✔ Urgency detection completed

✔ Models evaluated and validated

✔ Ready for integration & deployment

---