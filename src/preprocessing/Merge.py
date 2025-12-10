import pandas as pd

def merge_auto():
    df_cat = pd.read_csv("data/processed/labeled_categories.csv")
    df_urg = pd.read_csv("data/processed/labeled_urgency.csv")
    df_raw = pd.read_csv("data/interim/cleaned_emails.csv")

    # FIX: Convert cleaned_email to string for safe merge
    df_cat["cleaned_email"] = df_cat["cleaned_email"].fillna("").astype(str)
    df_urg["cleaned_email"] = df_urg["cleaned_email"].fillna("").astype(str)
    df_raw["cleaned_email"] = df_raw["cleaned_email"].fillna("").astype(str)

    # MERGE category + urgency
    merged = pd.merge(df_cat, df_urg, on=["id", "cleaned_email"], how="inner")

    # ADD raw_email from cleaned_emails.csv
    merged = pd.merge(merged, df_raw[["id", "raw_email"]], on="id", how="left")

    # SAVE FINAL DATASET
    merged.to_csv("data/processed/final_dataset.csv", index=False)

    print("🎉 final_dataset.csv created automatically!")

if __name__ == "__main__":
    merge_auto()
