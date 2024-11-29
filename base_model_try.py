import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

def clean_text(text):
    """Clean and validate text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return str(text).strip().lower()

def preprocess_data(english_texts, french_texts, max_len=20):
    """Improved preprocessing with start/end tokens and better tokenization."""
    # Clean and prepare texts
    english_texts = [clean_text(text) for text in english_texts]
    french_texts = ['<start> ' + clean_text(text) + ' <end>' for text in french_texts]
    
    # Filter out empty strings
    valid_indices = [i for i, (eng, fra) in enumerate(zip(english_texts, french_texts)) 
                    if eng and fra]
    english_texts = [english_texts[i] for i in valid_indices]
    french_texts = [french_texts[i] for i in valid_indices]
    
    if not english_texts or not french_texts:
        raise ValueError("No valid text pairs found after cleaning")
    
    # Initialize tokenizers with filters to keep important characters
    eng_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    fra_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    
    # Fit tokenizers
    eng_tokenizer.fit_on_texts(english_texts)
    fra_tokenizer.fit_on_texts(french_texts)
    
    # Convert texts to sequences
    eng_sequences = eng_tokenizer.texts_to_sequences(english_texts)
    fra_sequences = fra_tokenizer.texts_to_sequences(french_texts)
    
    # Pad sequences
    eng_padded = pad_sequences(eng_sequences, maxlen=max_len, padding='post')
    fra_padded = pad_sequences(fra_sequences, maxlen=max_len, padding='post')
    
    # Create decoder input and target data
    decoder_input = np.zeros_like(fra_padded)
    decoder_input[:, 1:] = fra_padded[:, :-1]
    decoder_target = fra_padded
    
    return (eng_padded, decoder_input, decoder_target, 
            eng_tokenizer, fra_tokenizer,
            len(eng_tokenizer.word_index) + 1,
            len(fra_tokenizer.word_index) + 1)

def build_basic_seq2seq(input_vocab_size, output_vocab_size, max_len=20):
    # Encoder
    encoder_inputs = Input(shape=(max_len,))
    enc_emb = Embedding(input_vocab_size, 256)(encoder_inputs)
    encoder = LSTM(256, return_state=True)
    encoder_outputs, state_h, state_c = encoder(enc_emb)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(max_len,))
    dec_emb = Embedding(output_vocab_size, 256)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True)
    decoder_outputs = decoder_lstm(dec_emb, initial_state=encoder_states)
    decoder_dense = Dense(output_vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def build_attention_seq2seq(input_vocab_size, output_vocab_size, max_len=20):
    # Encoder
    encoder_inputs = Input(shape=(max_len,))
    enc_emb = Embedding(input_vocab_size, 256)(encoder_inputs)
    encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    
    # Decoder
    decoder_inputs = Input(shape=(max_len,))
    dec_emb = Embedding(output_vocab_size, 256)(decoder_inputs)
    decoder_lstm = LSTM(256, return_sequences=True)
    decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    
    # Attention layer
    attention = Attention()
    attention_output = attention([decoder_outputs, encoder_outputs])
    
    # Combine attention output with decoder output
    decoder_concat = Concatenate()([decoder_outputs, attention_output])
    
    # Dense layers
    decoder_dense1 = Dense(256, activation='relu')(decoder_concat)
    decoder_outputs = Dense(output_vocab_size, activation='softmax')(decoder_dense1)
    
    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def translate_text(text, model_name="t5-small"):
    # Initialize the tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Add the correct task prefix to the text (for translation task)
    input_text = f"translate English to French: {text}"

    # Tokenize the input text and prepare it for the model
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Generate the translation using beam search
    summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True)

    # Decode the generated tokens into human-readable text
    translated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return translated_text

def train_and_evaluate_models(train_data_path, test_data_path=None, max_len=20, epochs=20, batch_size=64):
    try:
        # Load and validate training data
        print("Loading training data...")
        df_train = pd.read_csv(train_data_path)
        if 'english' not in df_train.columns or 'french' not in df_train.columns:
            raise ValueError("CSV must contain 'english' and 'french' columns")
        
        english_texts_train = df_train['english'].values
        french_texts_train = df_train['french'].values
        
        # Preprocess training data
        print("Preprocessing training data...")
        preprocessed_data_train = preprocess_data(english_texts_train, french_texts_train, max_len)
        if not preprocessed_data_train:
            raise ValueError("Training data preprocessing failed")
            
        (eng_padded_train, decoder_input_train, decoder_target_train,
         eng_tokenizer, fra_tokenizer,
         input_vocab_size, output_vocab_size) = preprocessed_data_train
        
        # If test data path is provided, load and preprocess test data
        if test_data_path:
            print("Loading test data...")
            df_test = pd.read_csv(test_data_path)
            if 'english' not in df_test.columns or 'french' not in df_test.columns:
                raise ValueError("Test CSV must contain 'english' and 'french' columns")
            
            english_texts_test = df_test['english'].values
            french_texts_test = df_test['french'].values
            
            # Preprocess test data
            print("Preprocessing test data...")
            preprocessed_data_test = preprocess_data(english_texts_test, french_texts_test, max_len)
            if not preprocessed_data_test:
                raise ValueError("Test data preprocessing failed")
            
            (eng_padded_test, decoder_input_test, decoder_target_test, _, _, _, _) = preprocessed_data_test
            
            # Split data for training and validation
            print("Splitting data...")
            splits_train = train_test_split(
                eng_padded_train, decoder_input_train, decoder_target_train,
                test_size=0.2, random_state=42
            )
            eng_train, eng_val, dec_input_train, dec_input_val, dec_target_train, dec_target_val = splits_train

            splits_test = train_test_split(
                eng_padded_test, decoder_input_test, decoder_target_test,
                test_size=0.2, random_state=42
            )
            eng_test, dec_input_test, dec_target_test = splits_test[0], splits_test[1], splits_test[2]
        
        else:  # If no test data file is provided, use training data for both
            print("No test data provided, using training data for validation.")
            eng_test, dec_input_test, dec_target_test = eng_padded_train, decoder_input_train, decoder_target_train
            eng_val, dec_input_val, dec_target_val = eng_train, dec_input_train, dec_target_train
        
        # Train models
        print("Training basic model...")
        basic_model = build_basic_seq2seq(input_vocab_size, output_vocab_size, max_len)
        basic_history = basic_model.fit(
            [eng_train, dec_input_train], dec_target_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([eng_val, dec_input_val], dec_target_val)
        )
        
        print("Training attention model...")
        attention_model = build_attention_seq2seq(input_vocab_size, output_vocab_size, max_len)
        attention_history = attention_model.fit(
            [eng_train, dec_input_train], dec_target_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([eng_val, dec_input_val], dec_target_val)
        )
        
        # Plot results
        plot_training_history(basic_history, attention_history)
        
        return basic_model, attention_model, eng_tokenizer, fra_tokenizer
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None, None, None

def plot_training_history(basic_history, attention_history):
    """Plot training history for both models."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(basic_history.history['loss'], label='Basic Model Training Loss')
    plt.plot(basic_history.history['val_loss'], label='Basic Model Validation Loss')
    plt.title('Basic Model Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(attention_history.history['loss'], label='Attention Model Training Loss')
    plt.plot(attention_history.history['val_loss'], label='Attention Model Validation Loss')
    plt.title('Attention Model Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

train_data_path = '/home/dl/train_translations_p.csv'
test_data_path = '/home/dl/test_translations_p.csv'

basic_model, attention_model, eng_tokenizer, fra_tokenizer = train_and_evaluate_models(train_data_path, test_data_path)

# Translate a sample sentence
sentence = "How are you?"
translated_sentence = translate_text(sentence)
print(f"Translated: {translated_sentence}")


