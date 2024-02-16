import os
import pickle
import re
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer, Bidirectional, Dense, LayerNormalization, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D, BatchNormalization, GRU, MultiHeadAttention
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from pygments.lexers import PythonLexer
from pygments import highlight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from tensorboard.plugins import projector
from datetime import datetime
import random
from sklearn.utils import shuffle
from typing import Dict, List, Optional, Set, Callable

import string
from tensorflow.keras.utils import get_file
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.manifold import TSNE
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
from tensorflow.keras.callbacks import Callback
from gensim.models import KeyedVectors
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
from gensim.scripts.glove2word2vec import glove2word2vec
import traceback
from typing import Optional, List, Set, Tuple
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.ops import summary_ops_v2
from io import StringIO
from tensorflow.keras.optimizers import Adam
import ast
import tensorflow_datasets as tfds
from transformers import AutoTokenizer, TFGPT2Model, GPT2Tokenizer, AutoModel
import requests
import zipfile
import tempfile


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python3'
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Dynamiczne zarządzanie wzrostem pamięci ustawione dla wszystkich GPU.")
    except RuntimeError as e:
        print(f"Błąd podczas ustawiania dynamicznego zarządzania wzrostem pamięci: {e}")


#os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/nvidia-cuda-toolkit/"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.mixed_precision.set_global_policy('float32')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

ZAPISZ_KATALOG = "mozgi"
KATALOG_LOGOW = "logs"
directory = "test"
log_dir = Path('logs')
tf.keras.backend.clear_session()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
loaded_model_name = ""
loaded_model = None
loaded_tokenizer = None
#tf.config.run_functions_eagerly(True)
# Nowe zmienne globalne
model_name = None
epoch = 0
step = 0

class TextProcessor:
    class PositionalEncoding(Layer):
        def __init__(self, d_model, **kwargs):
            super().__init__(**kwargs)
            self.d_model = d_model

        def get_angles(self, position, i):
            i = tf.cast(i, tf.float32)
            angles = 1 / tf.math.pow(10000.0, 2 * i / tf.cast(self.d_model, tf.float32))
            return angles

        def positional_encoding(self, position):
            angle_rads = self.get_angles(
                position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :],
            )
            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            return tf.cast(pos_encoding, tf.float32)

        def call(self, inputs):
            position = tf.shape(inputs)[1]
            pos_encoding = self.positional_encoding(position)
            return inputs + pos_encoding

    class WrappedMultiHeadAttention(Layer):
        def __init__(self, num_heads, d_model, rate=0.2, **kwargs):
            super().__init__(**kwargs)
            self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model, dropout=rate)

        def call(self, inputs):
            query, key, value = (tf.reshape(x, (-1, 1, x.shape[-1])) for x in inputs)
            return self.attention(query, key, value)

    class ACompleteTransformerBlock(Layer):
        def __init__(self, num_heads, d_model, dff, rate=0.2, **kwargs):
            super().__init__(**kwargs)
            self.attention = TextProcessor.WrappedMultiHeadAttention(num_heads, d_model, rate)
            self.ffn = self.point_wise_feed_forward_network(d_model, dff)
            self.layernorm1 = LayerNormalization(epsilon=1e-12)
            self.layernorm2 = LayerNormalization(epsilon=1e-12)
            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)
            self.pos_encoding = TextProcessor.PositionalEncoding(d_model)

        @staticmethod
        def point_wise_feed_forward_network(d_model, dff):
            return Sequential([
                Dense(dff, activation='relu'),
                Dense(d_model)
            ])

        def call(self, inputs, training):
            inputs = self.pos_encoding(inputs)
            attn_output = self.attention([inputs, inputs, inputs])
            out1 = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + out1)
            ffn_output = self.ffn(out1)
            out2 = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + out2)

    class TextGenerationCallback(Callback):
        def __init__(self, tokenizer, input_sequence_length, model_name, model, temperature=1.0):
            super().__init__()
            self.tokenizer = tokenizer
            self.input_sequence_length = input_sequence_length
            self.model_name = model_name
            self.model = model
            self.temperature = temperature
            self.generated_text_interval = 5  # Adjust the interval as desired
            self.seed_texts = ["Why is Python popular?", "What is AI?", "Explain neural networks", "Why is data important?"]
            self.current_seed_text_index = 0  # Added this line

        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.generated_text_interval == 0:
                # Use the current seed text and then move to the next one
                seed_text = self.seed_texts[self.current_seed_text_index]
                self.current_seed_text_index = (self.current_seed_text_index + 1) % len(self.seed_texts)  # Update the index
                generated_text = self.generate_text(seed_text, self.temperature, self.input_sequence_length)
                print(f"\nGenerated text from model '{self.model_name}' after epoch {epoch + 1}:\n{generated_text}\n")

        def generate_text(self, seed_texts, temperature=1.0, num_words=50):
            if isinstance(seed_texts, list) and len(seed_texts) > 0:
                seed_text = random.choice(seed_texts)
            else:
                seed_text = ""
            result = []
            len_predictions = len(self.tokenizer.word_index) + 1  # Add one for 0th index
            for _ in range(num_words):
                encoded_text = self.tokenizer.texts_to_sequences([seed_text])[0]
                padded_text = pad_sequences([encoded_text], maxlen=self.input_sequence_length, padding='pre')
                predictions = self.model.predict(padded_text)[0]
                valid_indices = np.where(~np.isnan(predictions))[0]
                valid_predictions = predictions[valid_indices]
                valid_predictions = np.where(np.isnan(valid_predictions), 1e-6, valid_predictions)
                try:
                    temp = float(temperature)
                except ValueError:
                    print(f"Error: Invalid temperature value: {temperature}")
                    temp = 1.0
                probabilities = np.exp(np.log(valid_predictions.clip(min=1e-6)) / temp)
                probabilities /= np.sum(probabilities)
                if valid_indices.size > 0 and not np.isnan(probabilities).all():
                    predicted_word_id = np.random.choice(valid_indices, p=probabilities)
                else:
                    predicted_word_id = np.random.choice(len_predictions)
                predicted_word = self.tokenizer.sequences_to_texts([[predicted_word_id]])[0]
                seed_text += ' ' + predicted_word
                result.append(predicted_word)
            return ' '.join(result)

    def __init__(
        self,
        directory: str,
        oov_token: str = '<OOV>',
        glove_file: str = None,

        gpt2_model_dir: str = 'gpt2-xl',
        model_name: str = None,
        input_sequence_length: int = 50,
        output_sequence_length: int = 50,
        batch_size: int = 32,
        lowercase: bool = False,
        handle_numbers: bool = True,
        handle_special_characters: bool = False,
        handle_stop_words: bool = True,
        lemmatize: bool = True,
        handle_python_code: bool = True,
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        learning_rate: float = 0.00001,
        amsgrad: bool = True,
        kernel_regularizer: float = 0.001,
        recurrent_regularizer: float = 0.001,
        bias_regularizer: float = 0.001,
        num_difficult_sequences: int = 50,
        stop_words: Optional[Set[str]] = None,
        log_dir: Optional[str] = 'logs',
    ):
        # Inicjalizacja atrybutów klasy
        self.oov_token = oov_token
        self.directory = directory
        self.glove_file = glove_file
        self.gpt2_model_dir = Path(gpt2_model_dir)
        self.model_name = model_name
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.batch_size = batch_size
        self.lowercase = lowercase
        self.handle_numbers = handle_numbers
        self.handle_special_characters = handle_special_characters
        self.handle_stop_words = handle_stop_words
        self.lemmatize = lemmatize
        self.handle_python_code = handle_python_code
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.amsgrad = amsgrad
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer
        self.num_difficult_sequences = num_difficult_sequences
        self.stop_words = set(stopwords.words('english')) if stop_words is None else stop_words
        self.tokenizer = None
        self.embedding_matrix = None
        self.vocab_size = 0
        self.processed_texts = []
        self.log_dir = log_dir
        self.glove_model = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None

        self.load_models()

    def create_tokenizer(self, texts: List[str]) -> None:
        if not texts:
            raise ValueError("Lista tekstów jest pusta lub None.")

        # Inicjalizacja tokenizera z domyślnymi tokenami GPT-2
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl", use_auth_token='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')

        # Dodanie dodatkowego tokena paddingu
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Nie dodawaj całych tekstów jako tokenów; zamiast tego użyj tokenizera do ich przetwarzania
        # Kodowanie i dekodowanie przykładowego tekstu dla weryfikacji
        for text in texts[:5]:  # Przetwarzaj tylko kilka przykładowych tekstów dla weryfikacji
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            encoded_text = self.tokenizer.encode(text, add_special_tokens=True)
            decoded_text = self.tokenizer.decode(encoded_text)
            print(f"Przykładowy zakodowany tekst: {encoded_text[:50]}")
            print(f"Przykładowy odkodowany tekst: {decoded_text[:50]}")

        print("Tokenizacja zakończona. Liczba unikalnych tokenów:", len(self.tokenizer.get_vocab()))

    def load_models(self):
        print("Loading GloVe model...")
        self.glove_model = self.load_glove_model()
        print("GloVe model loaded.")

        print("Loading GPT-2 model...")
        if not Path(self.gpt2_model_dir).exists():
            print(f"Model GPT-2 ({self.model_name}) nie jest dostępny lokalnie. Pobieranie...")
            self.gpt2_model = AutoModel.from_pretrained(self.model_name)
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Możesz rozważyć zapisanie modelu lokalnie, aby uniknąć ponownego pobierania w przyszłości
        else:
            self.load_gpt2_model()
        print("GPT-2 model loaded.")


    def download_file(self, url, save_path):
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def load_glove_model(self):
        glove_file = "glove.6B.100d.txt"
        if not os.path.exists(glove_file):
            print(f"Plik {glove_file} nie został znaleziony. Rozpoczynam pobieranie...")
            url = "http://nlp.stanford.edu/data/glove.6B.zip"
            with tempfile.NamedTemporaryFile(delete=False) as tmp_zip:
                self.download_file(url, tmp_zip.name)
                with zipfile.ZipFile(tmp_zip.name) as zf:
                    zf.extractall('.')
                    glove_file = 'glove.6B.100d.txt'

            print("Pobrano i wypakowano plik GloVe.")

        glove_model = {}
        with open(glove_file, 'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                embedding = np.array([float(val) for val in split_line[1:]])
                glove_model[word] = embedding

        return glove_model

    def load_gpt2_model(self):
        try:
            self.gpt2_model = AutoModel.from_pretrained(self.model_name, use_auth_token='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')
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token='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')
            print("Standardowy model GPT-2 załadowany pomyślnie.")
        except Exception as e:
            print(f"Błąd podczas wczytywania standardowego modelu GPT-2: {e}")

    def _load_external_datasets(self):
        datasets = ['webtext']
        for ds in datasets:
            for split in ['train', 'valid', 'test']:
                filename = f"{ds}.{split}.jsonl"
                if not os.path.exists(filename):  # Sprawdza, czy plik już istnieje
                    url = f"https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/{filename}"
                    print(f"Downloading {filename}...")
                    r = requests.get(url, stream=True)
                    if r.status_code == 200:
                        with open(filename, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"{filename} downloaded.")
                    else:
                        print(f"Failed to download {filename}. Status code: {r.status_code}")
                else:
                    print(f"{filename} already exists. Skipping download.")

    def PrepareTextData(self, text_data, tokenizer, sequence_length):
        """
        Przygotowuje dane tekstowe dla modelu.
        """
        # Zmodyfikowane sprawdzenie, czy tekst wejściowy nie jest pusty
        if text_data is None:
            raise ValueError("Tekst wejściowy jest None")
        if isinstance(text_data, np.ndarray) and text_data.size == 0:
            raise ValueError("Tekst wejściowy jest pustą tablicą NumPy")
        if isinstance(text_data, list) and not text_data:
            raise ValueError("Tekst wejściowy jest pustą listą")

        sequences = []
        for text in text_data:
            # Upewnij się, że tekst nie jest pusty
            if isinstance(text, str) and text:
                try:
                    encoded_text = tokenizer.encode(str(text), add_special_tokens=True)
                    # Uzupełnianie do długości sequence_length
                    if len(encoded_text) < sequence_length:
                        encoded_text += [0] * (sequence_length - len(encoded_text))
                    sequences.append(encoded_text[:sequence_length])
                except Exception as e:
                    print(f"Błąd podczas kodowania tekstu: {e}")
                    continue

        return sequences


    def preprocess_text(self, text):
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        if isinstance(text, list):
            text = ' '.join(text)  # Jeśli text jest listą, połącz go w jeden ciąg tekstowy
        words = word_tokenize(str(text))
        # Tymczasowo wyłącz filtrację, aby zobaczyć, czy wpływa to na wyniki
        # filtered_words = [word for word in words if word not in self.stop_words]
        if isinstance(text, str) and len(text) > 50:
            print(f"Przetworzony tekst: {text[:50]}...")  # Wyświetl fragment przetworzonego tekstu
        else:
            print(f"Przetworzony tekst: {text}")
        return ' '.join(words)

    def natural_questions_open(self):
        dataset_name = "natural_questions_open"
        print(f"Loading {dataset_name} dataset...")

        try:
            nq_open_data, nq_open_info = tfds.load(dataset_name, with_info=True)
            print("Dataset loaded.")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None, None

        train_data = nq_open_data["train"]

        def preprocess_fn(example):
            # Używaj tf.string, aby zachować kompatybilność z TensorFlow
            question = tf.py_function(self.preprocess_text, [example["question"]], Tout=tf.string)
            answer = tf.py_function(self.preprocess_text, [example["answer"]], Tout=tf.string)
            return question, answer

        processed_data = train_data.map(preprocess_fn)

        # Konwersja danych TensorFlow na format NumPy
        question_data, answer_data = [], []
        for q, a in processed_data.as_numpy_iterator():
            question_data.append(q.decode())  # Dekodowanie z formatu bytes
            answer_data.append(a.decode())

        print("Converting data to lists...")
        try:
            question_data, answer_data = zip(*[
                (tf.constant(q).numpy(), tf.constant(a).numpy()) for q, a in processed_data.as_numpy_iterator()
            ])
            print("Data converted to lists.")
        except Exception as e:
            print(f"Error during data conversion: {e}")
            return None, None

        print("Removing duplicates and generating unique data...")
        unique_data = set(zip(question_data, answer_data))
        unique_question_data, unique_answer_data = zip(*unique_data)
        print("Unique data generated.")

        print(f"Liczba wczytanych pytań: {len(question_data)}, liczba wczytanych odpowiedzi: {len(answer_data)}")
        return unique_question_data, unique_answer_data

    def generate_difficult_sequences(self, num_difficult_sequences: int, output_sequence_length: int) -> List[List[int]]:
        difficult_sequences = []

        if not self.processed_texts:
            print("Lista 'processed_texts' jest pusta.")
            return difficult_sequences

        for processed_text in self.processed_texts:
            difficult_sequence = []
            for token in processed_text:
                # Możesz zmienić te warunki, aby były mniej restrykcyjne
                if token and (token not in self.stop_words):
                    difficult_sequence.append(token)

            while len(difficult_sequence) < output_sequence_length:
                difficult_sequence.append(0)  # Może być konieczne dostosowanie tej logiki

            if difficult_sequence:
                difficult_sequences.append(difficult_sequence[:output_sequence_length])

            if len(difficult_sequences) >= num_difficult_sequences:
                break

        if not difficult_sequences:
            print("Nie udało się wygenerować trudnych sekwencji. Sprawdź dane i logikę generacji.")
        else:
            print(f"Udało się wygenerować {len(difficult_sequences)} trudnych sekwencji.")

        return difficult_sequences

    def create_sequences(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        batch_size: int,
        glove_file: str,
        directory: str,
        stop_words: Set[str],
        file_formats: Optional[List[str]] = ['.txt'],
        X_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        num_difficult_sequences: int = 50,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, int, int, np.ndarray, List[str], Dict[str, int]]:
        # Ładowanie wewnętrznych danych tekstowych
        processed_texts, word_counts = self._load_and_preprocess_files(directory, file_formats)

        # Ładowanie zewnętrznych zestawów danych
        self._load_external_datasets()
        external_texts = self.get_external_data()

        # Łączenie zewnętrznych danych z wewnętrznymi
        processed_texts.extend(external_texts)

        # Tworzenie tokenizatora i generowanie sekwencji
        self.create_tokenizer(processed_texts)

        # Uzyskanie maksymalnego indeksu tokena
        max_token_index = max(self.tokenizer.get_vocab().values())
        vocab_size = max_token_index + 1

        # Tworzenie macierzy embeddingów
        embedding_matrix = self.create_embedding_matrix(vocab_size)

        # Generowanie sekwencji
        sequences = self.generate_sequences(processed_texts, input_sequence_length)

        # Tworzenie danych treningowych i walidacyjnych
        X, y = self.create_X_y(sequences, input_sequence_length)

        # Dzielenie na zbiory treningowe i walidacyjne
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self.split_data(X, y, X_train, X_val, y_train, y_val)

        # Przygotowanie danych
        train_input_sequences, val_input_sequences, train_target_sequences, val_target_sequences = self.prepare_data(X_train, X_val, y_train, y_val, input_sequence_length, output_sequence_length)

        # Tworzenie zbiorów danych
        train_dataset, val_dataset = self.create_datasets(train_input_sequences, val_input_sequences, train_target_sequences, val_target_sequences, batch_size)

        # Debugowanie wygenerowanych zdań
        self.debug_generated_sentences(val_dataset, self.model)

        return (train_dataset, val_dataset, embedding_matrix, vocab_size, embedding_dim, difficult_sequences, processed_texts, word_counts)

    # Dodatkowe metody używane przez `create_sequences`:

    def get_external_data(self):
        # Pobieranie danych z 'natural_questions_open' i innych źródeł
        external_question_data, external_answer_data = self.natural_questions_open()
        return external_question_data + external_answer_data

    def create_embedding_matrix(self, vocab_size):
        # Tworzenie macierzy embeddingów
        embedding_dim = 100
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, idx in self.tokenizer.get_vocab().items():
            embedding_vector = self.glove_model.get(word)
            if embedding_vector is not None and idx < vocab_size:
                embedding_matrix[idx] = embedding_vector
        return embedding_matrix

    def generate_sequences(self, processed_texts, input_sequence_length):
        sequences = []
        for text in processed_texts:
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            elif isinstance(text, tf.Tensor):
                text = text.numpy().decode('utf-8')

            encoded_text = self.tokenizer.encode(text, add_special_tokens=True)
            if len(encoded_text) >= input_sequence_length:
                sequences.append(encoded_text[:input_sequence_length])
            else:
                # Dodanie paddingu do krótszych sekwencji
                sequences.append(encoded_text + [0] * (input_sequence_length - len(encoded_text)))

        if not sequences:
            raise ValueError("Nie udało się wygenerować żadnych sekwencji.")
        return sequences




    # W funkcji create_X_y
    def create_X_y(self, sequences, input_sequence_length):
        X, y = [], []
        for seq in sequences:
            for i in range(1, len(seq)):
                X_seq = seq[max(0, i - input_sequence_length):i]
                y_seq = seq[i]
                X.append(X_seq + [0] * (input_sequence_length - len(X_seq)))  # Dodanie paddingu
                y.append(y_seq)

        if not X or not y:
            raise ValueError("X lub y są puste. Brak danych do treningu.")

        return np.array(X), np.array(y)

    def split_data(self, X, y, X_train, X_val, y_train, y_val):
        if X_train is None or X_val is None or y_train is None or y_val is None:
            if len(X) == 0 or len(y) == 0:
                print("Dane treningowe są puste. Nie można podzielić pustych danych.")
                return np.array([]), np.array([]), np.array([]), np.array([])
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)

        return X_train, X_val, y_train, y_val

    def prepare_data(self, X_train, X_val, y_train, y_val, input_sequence_length, output_sequence_length):
        # Przygotowanie danych do treningu
        train_input_sequences = self.PrepareTextData(X_train, self.tokenizer, input_sequence_length)
        val_input_sequences = self.PrepareTextData(X_val, self.tokenizer, input_sequence_length)
        train_target_sequences = self.PrepareTextData(y_train, self.tokenizer, output_sequence_length)
        val_target_sequences = self.PrepareTextData(y_val, self.tokenizer, output_sequence_length)
        return train_input_sequences, val_input_sequences, train_target_sequences, val_target_sequences

    def create_datasets(self, train_input_sequences, val_input_sequences, train_target_sequences, val_target_sequences, batch_size):
        # Tworzenie zbiorów danych TensorFlow
        train_dataset = tf.data.Dataset.from_tensor_slices((train_input_sequences, train_target_sequences))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_input_sequences, val_target_sequences))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        return train_dataset, val_dataset

    def debug_generated_sentences(self, dataset, model):
        # Wypisywanie w konsoli wygenerowanych zdań dla debugowania
        for input_sequence, _ in dataset.take(5):
            prediction = model.predict(input_sequence.numpy())
            generated_text = self.tokenizer.decode(prediction[0])
            print(f"Wygenerowane zdanie: {generated_text}")

    def _load_and_preprocess_files(self, directory, file_formats):
        processed_texts = []
        word_counts = {}

        # Sprawdzenie, czy katalog istnieje
        print(f"Sprawdzanie ścieżki katalogu: {directory}")
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Błąd: Podana ścieżka '{directory}' nie jest katalogiem.")

        # Wczytanie plików z katalogu
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and any(f.endswith(format) for format in file_formats)]
        print(f"Znalezione pliki: {files}")
        if not files:
            raise FileNotFoundError("Brak plików w podanym formacie w katalogu.")

        for file in files:
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                lines = f.readlines()
                if not lines:
                    print(f"Plik {file} jest pusty.")
                    continue  # Pominięcie pustych plików

                for line in lines:
                    processed_line = self.preprocess_text(line)
                    processed_words = processed_line.split()  # Rozdzielanie tekstu na słowa
                    processed_texts.extend(processed_words)
                    word_count = len(processed_words)
                    word_counts[file] = word_counts.get(file, 0) + word_count
                print(f"Przetworzono plik: {file}, liczba słów: {word_count}")

        # Dodanie sprawdzenia, czy przetworzone teksty nie są puste
        if not processed_texts:
            raise ValueError("Brak przetworzonych tekstów. Proszę sprawdzić zawartość katalogu.")
        else:
            print(f"Liczba przetworzonych słów: {len(processed_texts)}")

        return processed_texts, word_counts

    def create_and_train_model(
        self,
        input_sequence_length,
        output_sequence_length,
        lstm_units,
        dropout_rate,
        epochs,
        batch_size,
        model_name,
        stop_words,
        learning_rate,
        amsgrad,
        kernel_regularizer,
        recurrent_regularizer,
        bias_regularizer,
        num_difficult_sequences,
        X_train,
        X_val,
        y_train,
        y_val,
        directory,
        glove_file,
        dtype=None
    ):
        (
            train_dataset,
            val_dataset,
            embedding_matrix,
            vocab_size,
            embedding_dim,
            difficult_sequences,
            processed_texts,
            word_counts,
        ) = self.create_sequences(
            input_sequence_length=input_sequence_length,
            output_sequence_length=output_sequence_length,
            stop_words=stop_words,
            batch_size=batch_size,
            glove_file=glove_file,
            directory=directory,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            num_difficult_sequences=num_difficult_sequences,
        )

        print("Wczytywanie danych pytań i odpowiedzi...")
        question_data, answer_data = self.natural_questions_open()

        assert isinstance(embedding_matrix, np.ndarray) and embedding_matrix.shape[1] == embedding_dim, "Parametr embedding_matrix powinien być tablicą NumPy o wymiarach (vocab_size, embedding_dim)"

        model = tf.keras.Sequential(name=model_name)
        model.add(tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=input_sequence_length,
            trainable=True
        ))
        model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="softmax"))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=3))
        model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation="softmax"))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            lstm_units,
            return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer),
            recurrent_regularizer=tf.keras.regularizers.l2(recurrent_regularizer),
            bias_regularizer=tf.keras.regularizers.l2(bias_regularizer),
        )))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
            lstm_units,
            return_sequences=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer),
            recurrent_regularizer=tf.keras.regularizers.l2(recurrent_regularizer),
            bias_regularizer=tf.keras.regularizers.l2(bias_regularizer),
        )))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=False,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(kernel_regularizer),
            recurrent_regularizer=tf.keras.regularizers.l2(recurrent_regularizer),
            bias_regularizer=tf.keras.regularizers.l2(bias_regularizer),
        )))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.BatchNormalization())
        model.add(TextProcessor.ACompleteTransformerBlock(name="ACompleteTransformerBlock", num_heads=8, d_model=256, dff=2048, rate=0.2))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        model.add(tf.keras.layers.Dense(embedding_dim + input_sequence_length))
        model.add(tf.keras.layers.Dense(2 * input_sequence_length))
        model.add(tf.keras.layers.Dense(input_sequence_length, dtype=dtype, activation="softmax"))

        loss = tf.keras.losses.CategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=amsgrad)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        log_dir = os.path.join(KATALOG_LOGOW, str(model_name))
        tensorboard_callbacks, log_dir = self.create_tensorboard_callback(model_name, train_dataset, epochs)
        early_stopping_callback = self.create_early_stopping_callback()
        checkpoint_callback = self.create_checkpoint_callback(model_name, log_dir)

        text_generation_callback = TextProcessor.TextGenerationCallback(self.tokenizer, input_sequence_length, model_name,model, temperature=1.0)
        callbacks = [
            tensorboard_callbacks,
            self.create_early_stopping_callback(),
            self.create_checkpoint_callback(model_name, log_dir)  # Teraz z log_dir
        ]
        print(model.summary())

        num_stages = 5
        stage_epochs = [int(epochs / num_stages)] * num_stages
        new_sequences = None
        train_input_sequences = None
        for stage in range(num_stages):
            stage_epochs_completed = sum(stage_epochs[:stage])
            print(f"Etap {stage + 1} - Szkolenie przez {stage_epochs[stage]} epoki...")
            if stage != 0:
                new_sequences = self.generate_difficult_sequences(
                    num_difficult_sequences,
                    output_sequence_length=output_sequence_length,
                )
                if new_sequences is not None and train_input_sequences is not None:
                    if new_sequences.shape[1] == train_input_sequences.shape[1]:
                        new_sequences = np.reshape(new_sequences, (new_sequences.shape[0], train_input_sequences.shape[1]))
                        train_input_sequences = np.concatenate((train_input_sequences, new_sequences))
                    else:
                        print("Ostrzeżenie: Kształt new_sequences nie pasuje do kształtu train_input_sequences. Pomijam konkatenację.")
                else:
                    print("Ostrzeżenie: new_sequences lub train_input_sequences jest None. Pomijam konkatenację.")

            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                initial_epoch=stage_epochs_completed,
                epochs=stage_epochs_completed + stage_epochs[stage],
                callbacks=callbacks
            )

            if stage < num_stages - 1:
                (
                    train_dataset,
                    val_dataset,
                    embedding_matrix,
                    vocab_size,
                    embedding_dim,
                    difficult_sequences,
                    processed_texts,
                    word_counts,
                ) = self.create_sequences(
                    input_sequence_length,
                    output_sequence_length,
                    stop_words,
                    batch_size,
                    glove_file,
                    directory,
                    X_train=X_train,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                    num_difficult_sequences=num_difficult_sequences,
                )

            print(f"Typ i kształt embedding_matrix po wypełnieniu: {type(embedding_matrix)}, {embedding_matrix.shape}")

        print("Ocena modelu...")
        accuracy, precision, recall, f1 = self.evaluate_model(model, X_val, y_val)
        print(f"Dokładność: {accuracy}, Precyzja: {precision}, Czułość: {recall}, F1-score: {f1}")

        metadata_file = os.path.join(log_dir, f"{model_name}-metadata.tsv")
        with open(metadata_file, "w") as f:
            for word, index in self.tokenizer.word_index.items():
                f.write(f"{word}\n")

        print(f"Model przeszkolony przez {epochs} epoki.")
        self.save_model_and_tokenizer(model, self.tokenizer, model_name)
        return (
            model,
            history,
            self.tokenizer,
            embedding_matrix,
            vocab_size,
            num_difficult_sequences,
            input_sequence_length,
            output_sequence_length,
            lstm_units,
            dropout_rate,
            epochs,
            batch_size,
            model_name,
            stop_words,
            learning_rate,
            amsgrad,
            kernel_regularizer,
            recurrent_regularizer,
            bias_regularizer,
            X_train,
            X_val,
            y_train,
            y_val,
            dtype,
        )

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        return accuracy, precision, recall, f1

    def create_tensorboard_callback(self, model_name, train_dataset, epoch):
        log_dir = os.path.join(self.log_dir, str(model_name))
        os.makedirs(log_dir, exist_ok=True)

        metadata_file = None
        if epoch == 0:
            metadata_file = os.path.join(log_dir, f'{model_name}-metadata.tsv')
            with tf.io.gfile.GFile(metadata_file, 'w') as f:
                for word, index in self.tokenizer.word_index.items():
                    f.write(f'{word}\n')

        # Add histogram for the transformer layer
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=2,
            write_graph=True,
            write_images=False,
            embeddings_freq=1,
            embeddings_metadata=metadata_file,
        )

        # Custom callback for recording loss and layer information
        class LossAndLayerInfoCallback(tf.keras.callbacks.Callback):
            def __init__(self, log_dir):
                super().__init__()
                self.log_dir = log_dir

            def on_epoch_end(self, epoch, logs=None):
                if self.model is not None:
                    # Record loss
                    if logs is not None and 'loss' in logs:
                        loss = logs['loss']
                        with tf.summary.create_file_writer(self.log_dir).as_default():
                            tf.summary.scalar('loss', loss, step=epoch)

                    # Record layer information for ACompleteTransformerBlock
                    layer_name = 'ACompleteTransformerBlock'  # Use the layer name as a string
                    tf.keras.utils.get_custom_objects()["ACompleteTransformerBlock"] = TextProcessor.ACompleteTransformerBlock
                    layer = self.model.get_layer(layer_name)
                    if layer is not None:
                        weights = layer.get_weights()
                        for i, weight in enumerate(weights):
                            with tf.summary.create_file_writer(self.log_dir).as_default():
                                tf.summary.histogram(f'{layer_name}_weight_{i}', weight, step=epoch)

        loss_and_layer_info_callback = LossAndLayerInfoCallback(log_dir)

        return [tensorboard_callback, loss_and_layer_info_callback], log_dir

    def create_checkpoint_callback(self, model_name, log_dir):
        if model_name is None:
            raise ValueError("model_name cannot be None")

        log_dir = os.path.join(self.log_dir, str(model_name))
        os.makedirs(log_dir, exist_ok=True)

        ckpt_filepath = os.path.join(log_dir, f'{model_name}-{{epoch:02d}}.ckpt')  # Save as .ckpt file
        h5_filepath = os.path.join(log_dir, f'{model_name}.h5')  # Save as .h5 file

        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_filepath,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                save_freq='epoch'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=h5_filepath,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                save_freq='epoch'
            )
        ]

    def create_early_stopping_callback(self):
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=10,
            verbose=1,
            mode='auto',
            baseline=None,
            restore_best_weights=True
        )

    def save_data(self, dataset, filename):
        """Save the dataset to a file using pickle."""
        with open(filename, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Data saved as {ZAPISZ_KATALOG}/{filename}')

    def save_tokenizer(self, tokenizer, filename):
        """Save the tokenizer to a file using pickle."""
        with open(filename, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Tokenizer saved as {ZAPISZ_KATALOG}/{filename}')

    def save_model_and_tokenizer(self, model, tokenizer, model_name):
        """Save the model weights and tokenizer to files."""
        model.save(f'{ZAPISZ_KATALOG}/{model_name}.h5')
        self.save_tokenizer(tokenizer, f'{ZAPISZ_KATALOG}/{model_name}_tokenizer.pkl')
        print("Model and tokenizer saved.")

    def load_model_and_tokenizer(self, model_file_name):
        """Load the model and tokenizer."""
        global loaded_model_name
        global loaded_model
        global loaded_tokenizer

        tokenizer_filename = f'{ZAPISZ_KATALOG}/{model_file_name}_tokenizer.pkl'
        if not os.path.exists(tokenizer_filename):
            print("Tokenizer file not found. Please create or load the model first.")
            return None, None, None

        loaded_model = tf.keras.models.load_model(f'{ZAPISZ_KATALOG}/{model_file_name}.h5')
        loaded_tokenizer = self.load_tokenizer(tokenizer_filename)
        loaded_model_name = model_file_name
        print("Model and tokenizer loaded.")
        return loaded_model, loaded_tokenizer

    def generate_code_ai(self, model_file_name, tokenizer, diversity):
        """Generate Python code using the AI model."""
        if model_file_name is None or tokenizer is None:
            print("Model or tokenizer not found. Please create or load the model first.")
            return

        num_words = int(input("Enter the number of words to generate: "))
        seed_text = input("Enter the seed text: ")
        model, _ = self.load_model_and_tokenizer(model_file_name)
        generated_text = self.generate_text(seed_text, model, tokenizer, num_words, diversity)
        print("Generated text:")
        print(generated_text)

    def generate_text(model, tokenizer, seed_text, max_length):
        result = []
        for _ in range(max_length):
            # Przetwarzamy tekst na sekwencję tokenów
            input_seq = tokenizer.texts_to_sequences([seed_text])[0]
            # Prognozujemy następny token
            predicted_token = model.predict_classes(np.array([input_seq]))[0]
            # Zamieniamy numer tokena na odpowiadające mu słowo
            predicted_word = None
            for word, index in tokenizer.word_index.items():
                if index == predicted_token:
                    predicted_word = word
                    break
            # Dodajemy przewidziane słowo do wynikowej sekwencji
            result.append(predicted_word)
            # Aktualizujemy seed_text, aby zawierał przewidziane słowo
            seed_text += " " + predicted_word
        return " ".join(result)


    def get_available_models():
        """
        Function that returns a list of available models in the current directory.
        32YetiPy-6b-23-200-128-03-0004-002-002-002
        """
        files = os.listdir()
        model_files = [file for file in files if file.endswith('.h5') and '_tokenizer' not in file]
        return model_files

def main():
    print("Witaj w AI Code Generator!")
    directory = "test"
    model = None
    tokenizer = None

    while True:
        print("\nCo chciałbyś zrobić?")
        print("1. Utwórz nowy model")
        print("2. Wczytaj istniejący model")
        print("3. Wygeneruj kod Pythona")
        print("4. Oceń model")
        print("5. Zakończ")
        choice = input("Podaj swój wybór (1-5): ")
        if choice == "1":
            glove_file = None,
            stop_words = set(stopwords.words("english"))
            model_name = input("Podaj nazwę modelu: ")
            input_sequence_length = 50  # Określ i przypisz wartość tutaj
            output_sequence_length = 50  # Określ i przypisz wartość tutaj
            batch_size = 32
            num_difficult_sequences = 50  # Zastąp odpowiednią wartością
            X_train = None
            X_val = None
            y_train = None
            y_val = None
            log_dir = KATALOG_LOGOW
            processor = TextProcessor(
                directory,
                oov_token="<OOV>",
                glove_file=glove_file,
                model_name=model_name,
                input_sequence_length=input_sequence_length,
                output_sequence_length=output_sequence_length,
                batch_size=batch_size,
                lowercase=False,
                handle_numbers=True,
                handle_special_characters=False,
                handle_stop_words=True,
                lemmatize=True,
                handle_python_code=True,
                lstm_units=128,
                dropout_rate=0.2,
                epochs=100,
                learning_rate=0.00001,
                amsgrad=True,
                kernel_regularizer=0.001,
                recurrent_regularizer=0.001,
                bias_regularizer=0.001,
                num_difficult_sequences=num_difficult_sequences,
                stop_words=stop_words,
                log_dir=log_dir,
            )

            dtype = tf.float32
            processor.create_and_train_model(
                directory="test",
                glove_file=glove_file,
                input_sequence_length=input_sequence_length,
                output_sequence_length=output_sequence_length,
                lstm_units=processor.lstm_units,
                dropout_rate=processor.dropout_rate,
                epochs=processor.epochs,
                batch_size=processor.batch_size,
                model_name=processor.model_name,
                stop_words=processor.stop_words,
                learning_rate=processor.learning_rate,
                amsgrad=processor.amsgrad,
                kernel_regularizer=processor.kernel_regularizer,
                recurrent_regularizer=processor.recurrent_regularizer,
                bias_regularizer=processor.bias_regularizer,
                num_difficult_sequences=num_difficult_sequences,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                dtype=dtype,
            )

            if model is not None:
                print("Model utworzony i wytrenowany pomyślnie!")

        elif choice == "2":
            model_files = get_available_models()
            if not model_files:
                print("Nie znaleziono modeli. Najpierw utwórz model.")
                continue

            print("Dostępne modele:")
            for i, model_file in enumerate(model_files):
                print(f"{i+1}. {model_file}")

            model_index = int(input("Podaj indeks modelu, który chcesz wczytać: ")) - 1
            model_file_name = model_files[model_index].split(".h5")[0]
            model, tokenizer = load_model_and_tokenizer(model_file_name)

        elif choice == "3":
            if model is None or tokenizer is None:
                print("Nie znaleziono modelu ani tokenizer. Najpierw utwórz lub wczytaj model.")
                continue

            generate_code_ai(model_file_name, tokenizer, diversity=0.5)

        elif choice == "4":
            if model is None or tokenizer is None:
                print("Nie znaleziono modelu ani tokenizer. Najpierw utwórz lub wczytaj model.")
                continue

            evaluate_model1(model, tokenizer, input_sequence_length)

        elif choice == "5":
            break

        else:
            print("Nieprawidłowy wybór. Wybierz liczbę od 1 do 5.")


if __name__ == "__main__":
    main()
