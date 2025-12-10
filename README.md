# AI-Smart-Email-Classifier
🚀 AI-Powered Smart Email Classifier — Project Progress (Current Stage)

This repository contains the initial completed foundation of the AI Email Classification System.
So far, the focus has been on project setup, dataset preparation, preprocessing, and dataset organization.

This README describes exactly what is completed right now—without mentioning any automatic loading, automatic labeling, or email generation features.

✅ Completed Work So Far
✔ 1. Project Folder Structure Setup

A clean machine-learning project layout has been created:

AI-Smart-Email-Classifier/
│
├── data/
│   ├── raw/          # Raw datasets
│   ├── interim/      # Cleaned data
│   ├── processed/    # Labeled/merged data
│   └── splits/       # Train/Test/Val files
│
├── src/
│   └── preprocessing/
│       ├── cleaner.py
│       ├── generate_cleaned_csv.py
│       ├── label_categories.py         (manual labeling if needed)
│       ├── label_urgency.py            (manual labeling if needed)
│       └── merge_datasets.py
│
└── README.md

✔ 2. Email Cleaning Pipeline Implemented

A preprocessing script (cleaner.py) has been developed that:

Removes HTML tags

Removes URLs and email addresses

Removes signatures (rule-based)

Converts text to lowercase

Removes unwanted characters

Normalizes spacing

Cleaned output is saved to:

data/interim/cleaned_emails.csv

✔ 3. Manual Category & Urgency Labeling Scripts Added

Scripts exist to allow manual labeling of emails for:

⭐ Category Labeling

(complaint / request / feedback / spam)

⭐ Urgency Labeling

(high / medium / low)

These scripts can be run as needed to create:

data/processed/labeled_categories.csv
data/processed/labeled_urgency.csv

✔ 4. Dataset Merge Script Implemented

A merging script combines:

Cleaned emails

Category labels

Urgency labels

Raw email text

Into a final consolidated dataset:

data/processed/final_dataset.csv

✔ 5. Train/Test/Validation Split Script Added

A script has been created to split the final dataset into:

data/splits/train.csv
data/splits/test.csv
data/splits/val.csv


This prepares the data for the upcoming model training stage.

📦 Files Successfully Generated So Far
File	Description
cleaned_emails.csv	Preprocessed emails ready for labeling
labeled_categories.csv	Category labels (manually created)
labeled_urgency.csv	Urgency labels (manually created)
final_dataset.csv	Combined dataset
train.csv	Training data split
test.csv	Test data split
val.csv	Validation data split
🧠 What’s Coming Next (Not Done Yet)

These tasks are NOT yet implemented:

❌ Model training (Category + Urgency)
❌ Transformers / BERT fine-tuning
❌ API development (FastAPI)
❌ Dashboard (Streamlit)
❌ Deployment (Docker / Cloud)

These will be added in the next stages of the project.

▶️ How to Run the Completed Parts
1. Clean the emails
python src/preprocessing/generate_cleaned_csv.py

2. Manually label categories
python src/preprocessing/label_categories.py

3. Manually label urgency
python src/preprocessing/label_urgency.py

4. Merge everything
python src/preprocessing/merge_datasets.py

5. Split into train/test/val
python src/preprocessing/split_dataset.py


📌 What’s Next (Not Done Yet)

These parts are not implemented yet but will be added later:

Category classification model training

Urgency classification model training

API development (FastAPI)

Dashboard (Streamlit)

Deployment (Docker / Cloud)