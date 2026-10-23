from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

def load_data():
    ''' Load and standardize the email and SMS datasets.'''
    email = pd.read_csv(RAW_DATA_DIR / "spam_ham_email.csv")
    sms = pd.read_csv(RAW_DATA_DIR / "spam_ham_sms.csv", encoding='latin-1')

    # Standardize email dataset
    email = email[['text', 'label']].copy()
    email['source'] = 'email'

    # Standardize
