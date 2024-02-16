import os
import pandas as pd
from datasets import load_dataset
import re
from transformers import AutoTokenizer, logging as transformers_logging
import ray
from tqdm.auto import tqdm
import torch
import nltk
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLineEdit, QTextEdit, QLabel
from PyQt6.QtCore import QThread, pyqtSignal

# Suppress transformers logging
transformers_logging.set_verbosity_error()

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Set device to GPU if available
if torch.cuda.is_available():
     device = torch.device("cuda:0") # Example: Use the first available GPU
else:
     device = torch.device("cpu")

ray.init(num_cpus=4)

stemmer = nltk.PorterStemmer()
stop_words = set(nltk.corpus.stopwords.words('english'))

def preprocess_code(code):
    # Remove comments
    code = re.sub(r'(#.*|\s*//.*|\s*/\*[\s\S]*?\*/)', '', code)
    # Remove white spaces
    code = re.sub(r'\s+', ' ', code)
    # Split joined identifiers
    code = re.sub('([a-z0-9])([A-Z])', r'\1 \2', code)
    # Convert to lower case
    code = code.lower()
    return code

def tokenize_code_functions(code_functions):
    tokenized_code_functions = []
    for function_name, docstring, label in code_functions:
        preprocessed_code = preprocess_code(function_name + " " + docstring)
        tokens = nltk.word_tokenize(preprocessed_code)
        # Remove stopwords and apply stemming
        tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.isalpha()]
        tokenized_code_functions.append((tokens, label))
    return tokenized_code_functions

@ray.remote
def preprocess_and_tokenize(dataset_name):
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split='train', cache_dir='./data_cache')  # Ustawienie lokalnej ścieżki cache
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def processing_function(examples):
        text_key = 'text' if 'text' in examples else next(iter(examples.keys()), 'text')
        processed_texts = []
        for text in examples.get(text_key, []):
            if isinstance(text, str):
                cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
            else:
                cleaned_text = str(text).lower()
            processed_texts.append(cleaned_text)
        examples[text_key] = processed_texts
        return tokenizer(examples[text_key], padding="max_length", truncation=True, return_tensors="pt")

    print(f"Processing dataset: {dataset_name}")
    processed_dataset = dataset.map(processing_function, batched=True, batch_size=1000)
    processed_data = {'text': [tokenizer.decode(ids[0], skip_special_tokens=True) for ids in processed_dataset['input_ids']]}
    return pd.DataFrame(processed_data)

def remove_duplicates(df):
    return df.drop_duplicates()

def split_dataset(df, train_frac=0.7, val_frac=0.15):
    train_dataset = df.sample(frac=train_frac, random_state=200)
    remaining = df.drop(train_dataset.index)
    validation_dataset = remaining.sample(frac=(val_frac / (1 - train_frac)), random_state=200)
    test_dataset = remaining.drop(validation_dataset.index)
    return train_dataset, validation_dataset, test_dataset

def generate_statistics(df):
    stats = {
        'Total samples': len(df),
        'Unique samples': df['text'].nunique(),
    }
    print(stats)

def save_datasets(train_dataset, val_dataset, test_dataset, dataset_name, base_path="processed_datasets"):
    try:
        os.makedirs(base_path, exist_ok=True)
        train_dataset.to_csv(os.path.join(base_path, f"{dataset_name}_train.csv"), index=False)
        val_dataset.to_csv(os.path.join(base_path, f"{dataset_name}_validation.csv"), index=False)
        test_dataset.to_csv(os.path.join(base_path, f"{dataset_name}_test.csv"), index=False)
        print(f"Datasets saved under {base_path} for {dataset_name}")
    except OSError as e:
        print(f"Error: {e}. Could not save datasets for {dataset_name}.")

def load_dataset_from_parquet(file_path):
    return pd.read_parquet(file_path)

def main():
    dataset_names = []
    print("Enter dataset names one by one. Type 'done' to finish:")
    while True:
        dataset_name = input("> ")
        if dataset_name.lower() == 'done':
            break
        dataset_names.append(dataset_name)


    futures = [preprocess_and_tokenize.remote(name) for name in dataset_names]
    results = [ray.get(future) for future in tqdm(futures, desc="Downloading and processing datasets")]

    for result, name in zip(results, dataset_names):
        combined_df = remove_duplicates(result)
        train_dataset, val_dataset, test_dataset = split_dataset(combined_df)
        generate_statistics(combined_df)
        save_datasets(train_dataset, val_dataset, test_dataset, name)

    print("Processing completed.")
    ray.shutdown()  # Optionally, shut down Ray explicitly when done

if __name__ == '__main__':
    main()
