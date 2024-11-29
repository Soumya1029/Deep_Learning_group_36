import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Attention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

def clean_text(text):
    """Clean and validate text data."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    return str(text).strip().lower()

def preprocess_data(english_texts, french_texts, max_len=20):
    # Clean and prepare texts
    english_texts = [clean_text(text) for text in english_texts]
    french_texts = [clean_text(text) for text in french_texts]
    
    # Filter out empty strings
    valid_indices = [i for i, (eng, fra) in enumerate(zip(english_texts, french_texts)) 
                    if eng and fra]
    english_texts = [english_texts[i] for i in valid_indices]
    french_texts = [french_texts[i] for i in valid_indices]
    
    if not english_texts or not french_texts:
        raise ValueError("No valid text pairs found after cleaning")
    
    # Initialize tokenizers
    eng_tokenizer = Tokenizer()
    fra_tokenizer = Tokenizer()
    
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

def translate_text(model, sentence, eng_tokenizer, fra_tokenizer, max_len=20):
    """Translate a single sentence using the given model."""
    try:
        # Clean and process input sentence
        sentence = clean_text(sentence)
        if not sentence:
            return "Error: Empty or invalid input"
        
        # Convert to sequence
        test_seq = eng_tokenizer.texts_to_sequences([sentence])
        test_input = pad_sequences(test_seq, maxlen=max_len, padding='post')
        
        # Generate translation
        output = model.predict([test_input, np.zeros((1, max_len))], verbose=0)
        
        # Decode prediction
        translation = " ".join([fra_tokenizer.index_word.get(np.argmax(token), "") 
                              for token in output[0] if np.argmax(token) != 0])
        
        return translation.strip()
    except Exception as e:
        return f"Translation error: {str(e)}"

def train_and_evaluate_models(data_path, max_len=20, epochs=10, batch_size=64):
    try:
        # Load and validate data
        print("Loading data...")
        df = pd.read_csv(data_path)
        if 'english' not in df.columns or 'french' not in df.columns:
            raise ValueError("CSV must contain 'english' and 'french' columns")
        
        english_texts = df['english'].values
        french_texts = df['french'].values
        
        # Preprocess data
        print("Preprocessing data...")
        preprocessed_data = preprocess_data(english_texts, french_texts, max_len)
        if not preprocessed_data:
            raise ValueError("Data preprocessing failed")
            
        (eng_padded, decoder_input, decoder_target,
         eng_tokenizer, fra_tokenizer,
         input_vocab_size, output_vocab_size) = preprocessed_data
        
        # Split data
        print("Splitting data...")
        splits = train_test_split(
            eng_padded, decoder_input, decoder_target,
            test_size=0.2, random_state=42
        )
        eng_train, eng_test, dec_input_train, dec_input_test, dec_target_train, dec_target_test = splits
        
        # Train models
        print("Training basic model...")
        basic_model = build_basic_seq2seq(input_vocab_size, output_vocab_size, max_len)
        basic_history = basic_model.fit(
            [eng_train, dec_input_train], dec_target_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([eng_test, dec_input_test], dec_target_test)
        )
        
        print("Training attention model...")
        attention_model = build_attention_seq2seq(input_vocab_size, output_vocab_size, max_len)
        attention_history = attention_model.fit(
            [eng_train, dec_input_train], dec_target_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([eng_test, dec_input_test], dec_target_test)
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
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(attention_history.history['loss'], label='Attention Model Training Loss')
    plt.plot(attention_history.history['val_loss'], label='Attention Model Validation Loss')
    plt.title('Attention Model Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # Configuration
        data_path = '/home/dl/combined_vocab.csv'  # Update this path
        max_len = 20
        epochs = 10
        batch_size = 64
        
        # Train models
        models = train_and_evaluate_models(data_path, max_len, epochs, batch_size)
        if not all(models):
            raise ValueError("Model training failed")
            
        basic_model, attention_model, eng_tokenizer, fra_tokenizer = models
        
        # Test translations
        test_sentences = [
            "Hello, how are you?",
            "I love programming",
            "What time is it?"
        ]
        
        print("\nTest Translations:")
        for sentence in test_sentences:
            print(f"\nOriginal: {sentence}")
            print(f"Basic Model: {translate_text(basic_model, sentence, eng_tokenizer, fra_tokenizer)}")
            print(f"Attention Model: {translate_text(attention_model, sentence, eng_tokenizer, fra_tokenizer)}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()
