from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# File paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


def load_data():
    ''' Load and standardize the email and SMS datasets.'''
    email = pd.read_csv(RAW_DATA_DIR / "spam_ham_email.csv")
    sms = pd.read_csv(RAW_DATA_DIR / "spam_ham_sms.csv", encoding='latin-1')

    # Standardize email dataset
    email = email[['text', 'label']].copy()
    email['source'] = 'email'

    # Standardize SMS dataset
    sms = sms[['v2', 'v1']].copy()
    sms.columns = ['text', 'label']
    sms['source'] = 'sms'

    # Combine datasets
    df = pd.concat([email, sms], ignore_index=True)

    return df


def clean_data(df):
    '''Remove invalid, conflicting, and duplicate messages.'''
    df = df.copy()

    # Remove missing values
    df = df.dropna(subset=['text', 'label', 'source'])

    # Remove empty messages
    df = df[df['text'].str.strip() != '']

    # Find messages with conflicting labels
    label_counts = df.groupby(['text', 'source'])['label'].nunique()
    conflicting_pairs = label_counts[label_counts > 1].index

    # Remove conflicting messages
    row_pairs = pd.MultiIndex.from_frame(df[['text', 'source']])
    df = df[~row_pairs.isin(conflicting_pairs)]

    # Remove duplicate messages
    df = df.drop_duplicates(subset=['text', 'source'])

    # Keep only necessary columns
    df = df[['text', 'label', 'source']]

    return df.reset_index(drop=True)


def split_data(df):
    """Create training and testing sets without text leakage."""

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=123
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df["text"]
        )
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    return train_df, test_df


def save_data(train_df, test_df):
    """Save processed datasets as CSV files."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        PROCESSED_DATA_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DATA_DIR / "test.csv",
        index=False
    )


def main():
    df = load_data()

    assert set(df["label"].dropna()) == {"ham", "spam"}
    assert set(df["source"].dropna()) == {"email", "sms"}

    print(f"Original messages: {len(df)}")

    # Check raw labels and sources
    print("\nOriginal labels:")
    print(df["label"].value_counts(dropna=False))

    print("\nOriginal sources:")
    print(df["source"].value_counts(dropna=False))

    df = clean_data(df)

    print(f"Cleaned messages: {len(df)}")

    train_df, test_df = split_data(df)
    
    test_ratio = len(test_df) / len(df)
    print(f"\nActual test proportion: {test_ratio:.2%}")

    # Verify no identical text appears in both datasets
    overlap = set(train_df["text"]) & set(test_df["text"])
    assert len(overlap) == 0, "Data leakage detected!"

    # Verify both datasets contain valid labels and sources
    assert set(df["label"]) == {"ham", "spam"}
    assert set(df["source"]) == {"email", "sms"}

    # Check both train and test sets
    for dataset in [train_df, test_df]:
        assert set(dataset["label"]) == {"ham", "spam"}
        assert set(dataset["source"]) == {"email", "sms"}

    save_data(train_df, test_df)

    print(f"\nTraining messages: {len(train_df)}")
    print(f"Testing messages: {len(test_df)}")

    print("\nTraining class distribution:")
    print(train_df["label"].value_counts(normalize=True))

    print("\nTesting class distribution:")
    print(test_df["label"].value_counts(normalize=True))

    print("\nTraining source distribution:")
    print(train_df["source"].value_counts(normalize=True))

    print("\nTesting source distribution:")
    print(test_df["source"].value_counts(normalize=True))

    # Compare spam rates within each source
    for name, dataset in [("Train", train_df), ("Test", test_df)]:
        print(f"\n{name} source/class distribution:")
        print(
            pd.crosstab(
                dataset["source"],
                dataset["label"],
                normalize="index"
            )
        )


    print("\nProcessed datasets saved successfully.")


if __name__ == "__main__":
    main()
