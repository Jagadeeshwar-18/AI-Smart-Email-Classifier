# AI-Smart-Email-Classifier
🚀 AI-Powered Smart Email Classifier — Project Progress

This repository contains the **core implementation** of an AI-powered email classification system for enterprises.  
The project focuses on **email preprocessing, dataset preparation, model training, and evaluation**.

This README reflects **what has been implemented so far**, along with clearly defined next steps.

---

## ✅ Completed Work So Far

### ✔ 1. Project Folder Structure Setup

A clean, scalable machine-learning project structure has been created:

```text
AI-Smart-Email-Classifier/
│
├── data/
│ ├── raw/ # Raw datasets
│ ├── interim/ # Cleaned emails
│ ├── processed/ # Labeled & merged datasets
│ └── splits/ # Train/Test/Validation splits
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
│ ├── train_logistic_regression.py
│ ├── train_naive_bayes.py
│ ├── train_distilbert.py
│ ├── evaluate_distilbert.py
│ └── label_mapping.py
│
├── models/
│ └── category/
│ ├── logistic_regression.pkl
│ ├── naive_bayes.pkl
│ └── distilbert/
│
└── README.md
```

---

### ✔ 2. Email Cleaning Pipeline Implemented

The preprocessing pipeline performs:

- Removal of HTML tags  
- Removal of URLs and email addresses  
- Signature removal (rule-based)  
- Lowercasing  
- Removal of special characters  
- Whitespace normalization  

Cleaned output is saved as:
data/interim/cleaned_emails.csv

---

### ✔ 3. Manual Category & Urgency Labeling Support

Scripts are available for **manual dataset labeling**:

#### Category Labels
- complaint  
- request  
- feedback  
- spam  

#### Urgency Labels
- high  
- medium  
- low  

Generated files:
data/processed/labeled_categories.csv
data/processed/labeled_urgency.csv


---

### ✔ 4. Dataset Merging & Splitting

A merged dataset is created containing:

- Raw email text  
- Cleaned email text  
- Category labels  
- Urgency labels  

Final dataset:
data/processed/final_dataset.csv


Train/Test/Validation splits:
data/splits/train.csv
data/splits/test.csv
data/splits/val.csv


---

## 🤖 Model Development (Implemented)

### ✔ 5. Baseline Email Categorization Models

Two baseline machine learning models were trained using **TF-IDF features**:

- Logistic Regression  
- Naive Bayes  

These models serve as performance benchmarks for transformer-based models.

**Baseline Performance (Test Set):**
- Accuracy ≈ 92%
- Macro F1-score ≈ 92%

Saved models:
models/category/logistic_regression.pkl
models/category/naive_bayes.pkl


---

### ✔ 6. Transformer-Based Email Categorization (DistilBERT)

A **DistilBERT** model was fine-tuned for multi-class email categorization.

- Training performed for 1 epoch due to computational constraints
- Evaluated on a held-out test set

**DistilBERT Test Results:**
- Accuracy: **94.17%**
- Macro F1-score: **94.53%**

Class-wise performance showed strong results across all categories, with particularly high precision for spam detection and high recall for request classification.

Saved model:
models/category/distilbert/


---

## 📊 Files Generated So Far

| File | Description |
|-----|-------------|
| cleaned_emails.csv | Preprocessed email text |
| labeled_categories.csv | Category labels |
| labeled_urgency.csv | Urgency labels |
| final_dataset.csv | Combined dataset |
| train.csv | Training split |
| test.csv | Test split |
| val.csv | Validation split |
| logistic_regression.pkl | Baseline model |
| naive_bayes.pkl | Baseline model |
| distilbert/ | Fine-tuned transformer model |

---

## ▶️ How to Run the Implemented Pipeline

### 1. Clean emails
```bash
python src/preprocessing/generate_cleaned_csv.py
```

### 2. Label categories
```bash
python src/preprocessing/label_categories.py
```
### 3. Label urgency
```bash
python src/preprocessing/label_urgency.py
```
### 4. Merge datasets
```bash
python src/preprocessing/merge_datasets.py
```
### 5. Split data
```bash
python src/preprocessing/split_dataset.py
```
### 6. Train baseline models
```bash
python src/models/train_logistic_regression.py
python src/models/train_naive_bayes.py
```
### 7. Train DistilBERT
```bash
python src/models/train_distilbert.py
```
### 8. Evaluate DistilBERT
```bash
python src/models/evaluate_distilbert.py
```