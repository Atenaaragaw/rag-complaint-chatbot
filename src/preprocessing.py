import pandas as pd
import re
import os

def clean_text(text):
    if pd.isna(text):
        return ""
    # Remove CFPB redaction marks (XXXX)
    text = re.sub(r'X+', '', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def run_preprocessing(input_path, output_path):
    print("--- Loading Dataset ---")
    # Load only necessary columns to save memory
    cols = ['Complaint ID', 'Product', 'Consumer complaint narrative', 'Issue', 'Sub-issue', 'Company', 'State', 'Date received']
    df = pd.read_csv(input_path, usecols=cols)

    # 1. Filter Products
    target_products = ['Credit card', 'Personal loan', 'Savings account', 'Money transfers']
    # Use str.contains for flexibility (e.g., "Credit card or prepaid card")
    df = df[df['Product'].str.contains('|'.join(target_products), case=False, na=False)]

    # 2. Filter missing narratives
    df = df.dropna(subset=['Consumer complaint narrative'])
    
    print(f"Filtered to {len(df)} relevant complaints.")

    # 3. Clean Narratives
    print("--- Cleaning Text (this may take a minute) ---")
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)

    # 4. Save processed data
    df.to_csv(output_path, index=False)
    print(f"Success! Processed data saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing('data/raw/complaints.csv', 'data/processed/filtered_complaints.csv')