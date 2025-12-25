# AI-Smart-Email-Classifier

---

## 🚀 Project Title
- **AI-Powered Smart Email Classifier for Enterprises**

---

## 📌 Problem Statement
- Enterprises receive large volumes of emails daily
- Manual email triaging is slow and inefficient
- Critical emails may be delayed
- Automation is required for:
  - Email categorization
  - Urgency prioritization

---

## 🎯 Project Objectives
- Automatically classify emails into categories
- Detect urgency levels for prioritization
- Reduce manual workload
- Improve enterprise response times

---

## 🧠 Core Capabilities
- Natural Language Processing (NLP)
- Machine Learning & Transformer Models
- Hybrid Rule + ML Decision Systems
- Scalable enterprise-ready design

---

## 🏗️ System Architecture
```text
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
```

---

## ✅ Implemented Milestones

### ✔ Milestone 1: Data Preparation & Preprocessing

- Email cleaning pipeline (HTML, URLs, signatures removed)
- Manual labeling support for category and urgency
- Dataset merging and train/test/validation splits

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

**DistilBERT Test Results**
- Accuracy: **94.17%**
- Macro F1-score: **94.53%**

---

### ✔ Milestone 3: Urgency Detection Module

A hybrid urgency detection system was implemented.

#### Rule-Based Detection
- Captures explicit urgency signals (e.g., *ASAP*, *system down*, *urgent*)

#### ML-Based Detection
- Logistic Regression with TF-IDF features
- Predicts: **High / Medium / Low**

**Urgency Model Performance**
- Accuracy: **95%**
- Weighted F1-score: **0.95**
- Macro F1-score: **0.91**

#### Hybrid Decision Logic
- Rule-based **HIGH** urgency overrides ML predictions
- ML handles nuanced cases

---

## 📊 Results Summary

| Task                  | Model                  | Accuracy | Macro F1 |
|-----------------------|------------------------|----------|----------|
| Email Categorization  | Logistic Regression    | ~92%     | ~92%     |
| Email Categorization  | DistilBERT             | 94.17%   | 94.53%   |
| Urgency Detection     | Hybrid (Rules + ML)    | 95%      | 0.91     |

---

## 📁 Project Structure
```text
AI-Smart-Email-Classifier:
  data:
    raw:
    interim:
    processed:
    splits:
  src:
    preprocessing:
      cleaner.py
      generate_cleaned_csv.py
      label_categories.py
      label_urgency.py
      merge_datasets.py
      split_dataset.py
    models:
      vectorizer.py
      label_mapping.py
      train_logistic_regression.py
      train_naive_bayes.py
      train_distilbert.py
      evaluate_distilbert.py
      train_urgency_model.py
      urgency_rules.py
      hybrid_urgency.py
  models:
    category:
    urgency:
  README.md
  .gitignore
```

> **Note:** Large datasets and trained models are excluded from version control.

---

## ▶️ How to Run


# Preprocess emails
```bash
python src/preprocessing/generate_cleaned_csv.py
```
# Split dataset
```bash
python src/preprocessing/split_dataset.py
```
# Train category models
```bash
python src/models/train_logistic_regression.py
python src/models/train_distilbert.py
```
# Train urgency model
```bash
python src/models/train_urgency_model.py
```
## 🧪 Hybrid Urgency Logic
- Rule-based **HIGH** urgency overrides ML prediction
- ML model handles remaining cases
- Ensures high precision for critical emails

---

## 🚧 Excluded from Version Control
- Raw datasets
- Processed datasets
- Trained models
- Virtual environments

---

## 🚀 Future Enhancements
- FastAPI backend
- Streamlit dashboard
- Docker-based deployment
- Cloud hosting (AWS / Azure / GCP)

---

## 🎯 Project Status
- ✔ Data pipeline completed
- ✔ Email categorization completed
- ✔ Urgency detection completed
- ✔ Models evaluated and validated
- ✔ Ready for integration & deployment
