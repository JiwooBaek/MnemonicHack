"""
DL-SCA for Mnemonic Recovery
python train.py
"""

import os
import re
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization, Activation,
    Dropout, LSTM, Embedding, Concatenate, TimeDistributed, Dense,
    Bidirectional, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau, Callback
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from tqdm.auto import tqdm

# ------------------------------------------------------------------------------
# Configuration & Hyperparameters
# ------------------------------------------------------------------------------
# Hardware optimization
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Paths
WORDLIST_FILE = "./bip39_english.c"
MODEL_SAVE_DIR = "./models"
LOG_CSV_DIR = "./logs_manual_prep"

# Input Data Paths (Tuple: Path, Label Group ID, Invert Signal Flag)
TRACE_GROUPS = [
    ("../datasets/traces/filtered_trace_A_common_20000.npy", 0, False),
    ("../datasets/traces/filtered_trace_B_common_20000.npy", 0, False),
    ("../datasets/traces/filtered_trace_A_a_20000.npy", 1, False),
    ("../datasets/traces/filtered_trace_B_b_20000.npy", 2, False),
    ("../datasets/traces/filtered_trace_C_c_20000.npy", 3, False),
]

MNEMONIC_FILES = [
    "../datasets/keys/mnemonic_code_common_20000.txt",
    "../datasets/keys/mnemonic_code_a_20000.txt",
    "../datasets/keys/mnemonic_code_b_20000.txt",
    "../datasets/keys/mnemonic_code_c_20000.txt",
]

# Model Parameters
DECODER_STEPS = 24
ROI_START = 70000
WINDOW_LENGTH = 128000
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
JITTER_MAX = 1000

# Create directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_CSV_DIR, exist_ok=True)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def load_bip39_wordlist_from_c(filepath: str):
    """Parses C source file to extract BIP39 English wordlist."""
    with open(filepath, "r", encoding="utf-8") as f:
        src = f.read()
    
    # Extract array content
    m = re.search(r"BIP39_WORDLIST_ENGLISH\s*\[.*?\]\s*=\s*\{", src)
    if not m: 
        raise RuntimeError("Target array (BIP39_WORDLIST_ENGLISH) not found.")
    
    start, end = m.end(), src.find("};", m.end())
    words = re.findall(r'"([a-z]+)"', src[start:end])
    return words[:2048]

def load_mnemonic_labels(txt_path: str, word_to_idx: dict, max_words=24):
    """Converts mnemonic text files to integer index arrays."""
    labels = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            words = line.split()
            if len(words) != max_words: continue
            
            labels.append([word_to_idx[w] for w in words])
            
    return np.array(labels, dtype=np.int32)


# ------------------------------------------------------------------------------
# Data Generator
# ------------------------------------------------------------------------------
class SCADataGenerator(Sequence):
    """
    Handles data loading, jitter augmentation, and batching.
    Returns: ({waveform, position}, labels)
    """
    def __init__(self, metadata, arrays, batch_size=32, dim=128000, 
                 n_channels=1, seq_len=24, augment=False, roi_start=70000, jitter_max=1000):
        self.metadata = metadata
        self.arrays = arrays  # List of memmapped arrays
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.augment = augment 
        self.roi_start = roi_start
        self.jitter_max = jitter_max
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.metadata) / self.batch_size))

    def __getitem__(self, index):
        batch_meta = self.metadata[index * self.batch_size:(index + 1) * self.batch_size]
        X_wave, X_pos, y = self.__data_generation(batch_meta)
        return {"waveform_input": X_wave, "position_input": X_pos}, y

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.metadata)

    def __data_generation(self, batch_meta):
        X_wave = np.empty((self.batch_size, self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.seq_len), dtype=np.int32)
        X_pos = np.tile(np.arange(self.seq_len), (self.batch_size, 1)).astype(np.int32)

        for i, (g_idx, t_idx, label, _) in enumerate(batch_meta):
            raw_full = self.arrays[g_idx][t_idx]
            
            # Jitter augmentation
            current_start = self.roi_start
            if self.augment:
                shift = np.random.randint(-self.jitter_max, self.jitter_max + 1)
                current_start += shift
            
            # Boundary checks
            current_start = max(0, min(current_start, len(raw_full) - self.dim))
            segment = raw_full[current_start : current_start + self.dim].copy()
            
            # Standardization
            mean = np.mean(segment)
            std = np.std(segment)
            segment = (segment - mean) / (std if std > 1e-8 else 1.0)
            
            # Scale augmentation
            if self.augment:
                scale = np.random.uniform(0.9, 1.1)
                segment *= scale

            X_wave[i] = np.expand_dims(segment, axis=-1)
            y[i] = label

        return X_wave, X_pos, y


# ------------------------------------------------------------------------------
# Custom Callbacks
# ------------------------------------------------------------------------------
class TqdmValLogger(Callback):
    """Custom TQDM progress bar with validation metrics logging."""
    def __init__(self):
        self.pbar = None

    def on_epoch_begin(self, epoch, logs=None):
        total_steps = self.params.get('steps')
        self.pbar = tqdm(
            total=total_steps, 
            desc=f"Epoch {epoch + 1:03d}", 
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            leave=False
        )

    def on_train_batch_end(self, batch, logs=None):
        if self.pbar: self.pbar.update(1)

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar: self.pbar.close()
        logs = logs or {}
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')
        lr = logs.get('lr')
        
        # Fallback for LR retrieval
        if lr is None:
            if hasattr(self.model.optimizer, 'lr'):
                lr = K.get_value(self.model.optimizer.lr)
            elif hasattr(self.model.optimizer, 'learning_rate'):
                 lr = K.get_value(self.model.optimizer.learning_rate)
        
        print(f"Epoch {epoch + 1:03d} | val_loss: {val_loss:.4f} | "
              f"val_accuracy: {val_acc:.4f} | lr: {lr:.2e}")


# ------------------------------------------------------------------------------
# Model Architecture
# ------------------------------------------------------------------------------
def build_table_spec_model(input_len, num_words, seq_len=24):
    """
    Encoder-Decoder architecture with Attention.
    Encoder: CNN blocks + Bi-LSTM.
    Decoder: LSTM + Dot-product Attention.
    """
    # Encoder
    inp_wave = Input(shape=(input_len, 1), name='waveform_input')
    
    x = Conv1D(32, 11, strides=2, padding='same')(inp_wave)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x) 
    
    x = Conv1D(64, 11, strides=2, padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(128, 9, strides=2, padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Conv1D(256, 7, strides=2, padding='same')(x)
    x = BatchNormalization()(x); x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = Dropout(0.3)(x)
    encoder_output = Bidirectional(LSTM(128, return_sequences=True))(x)

    # Decoder
    inp_pos = Input(shape=(seq_len,), name='position_input')
    pos_emb = Embedding(input_dim=seq_len, output_dim=256)(inp_pos)
    decoder_lstm_out = LSTM(256, return_sequences=True)(pos_emb)
    
    # Attention
    attention_layer = Attention(score_mode='dot')
    context_vector = attention_layer([decoder_lstm_out, encoder_output])
    
    concat_out = Concatenate()([decoder_lstm_out, context_vector])
    
    # Output
    out = TimeDistributed(Dense(num_words, activation='softmax', dtype='float32'), name='word_output')(concat_out)
    
    return Model(inputs=[inp_wave, inp_pos], outputs=out)


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Compute dtype: {policy.compute_dtype}")
    
    # 1. Load Data
    bip39_words = load_bip39_wordlist_from_c(WORDLIST_FILE)
    word_to_idx = {w: i for i, w in enumerate(bip39_words)}
    NUM_KEYS = len(bip39_words)
    
    all_label_sets = [load_mnemonic_labels(f, word_to_idx) for f in MNEMONIC_FILES]
    
    loaded_arrays = []
    for path, _, _ in TRACE_GROUPS:
        if os.path.exists(path):
            loaded_arrays.append(np.load(path, mmap_mode='r'))
        else:
            print(f"[!] File not found: {path}")
            loaded_arrays.append(None)

    # 2. Partition Data (Train/Val)
    train_metadata = []
    val_metadata = []

    # Mapping: 0-3 -> Train (Devices A, B), 4 -> Val (Device C)
    for g_idx, (path, label_set_idx, is_c) in enumerate(TRACE_GROUPS):
        if loaded_arrays[g_idx] is None: continue
        
        num_traces = len(loaded_arrays[g_idx])
        labels = all_label_sets[label_set_idx]
        
        # Sync trace count with labels
        num_traces = min(num_traces, len(labels))
        
        group_meta = [(g_idx, t_idx, labels[t_idx], is_c) for t_idx in range(num_traces)]
        
        if g_idx in [0, 1, 2, 3]:
            train_metadata.extend(group_meta)
        elif g_idx == 4:
            val_metadata.extend(group_meta)

    print(f"Training samples: {len(train_metadata)}")
    print(f"Validation samples: {len(val_metadata)}")

    # 3. Initialize Generators
    train_gen = SCADataGenerator(
        train_metadata, loaded_arrays, 
        batch_size=BATCH_SIZE, seq_len=DECODER_STEPS, 
        augment=True, roi_start=ROI_START, jitter_max=JITTER_MAX
    )

    val_gen = SCADataGenerator(
        val_metadata, loaded_arrays, 
        batch_size=BATCH_SIZE, seq_len=DECODER_STEPS, 
        augment=False, roi_start=ROI_START
    )

    # 4. Build and Train Model
    model = build_table_spec_model(WINDOW_LENGTH, NUM_KEYS, DECODER_STEPS)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_DIR, "best_model.keras"), 
            save_best_only=True, monitor='val_accuracy', mode='max', verbose=0
        ),
        CSVLogger(os.path.join(LOG_CSV_DIR, "training_log.csv"), append=True),
        ReduceLROnPlateau(
            monitor='val_accuracy', mode='max', factor=0.5, 
            patience=15, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy', mode='max', patience=20, 
            restore_best_weights=True
        ),
        TqdmValLogger()
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0  # Handled by TqdmValLogger
    )
    
    print("Training complete.")