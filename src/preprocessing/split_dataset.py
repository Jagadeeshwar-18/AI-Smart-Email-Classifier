import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    df = pd.read_csv("data/processed/final_dataset.csv")

    # Train (80%) + Temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)

    # Temp becomes val (10%) + test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, shuffle=True)

    # Save the splits
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv", index=False)
    test_df.to_csv("data/splits/test.csv", index=False)

    print("🎉 Dataset successfully split into train, val, and test!")

if __name__ == "__main__":
    split_dataset()
