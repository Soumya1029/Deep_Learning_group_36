import pandas as pd
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 1: Read Data from Files
# Change these paths to the correct locations of your English and French text files
with open('/home/dl/small_vocab_en.csv', 'r', encoding='utf-8') as f:
    english_text = f.readlines()

with open('/home/dl/small_vocab_fr.csv', 'r', encoding='utf-8') as f:
    french_text = f.readlines()

# Remove any leading/trailing whitespace characters, like newline characters
english_text = [line.strip() for line in english_text]
french_text = [line.strip() for line in french_text]

# Ensure both files have the same number of lines
assert len(english_text) == len(french_text), "Mismatch in number of lines between files"

# Step 2: Create a DataFrame for easy handling
df = pd.DataFrame({
    'english': english_text,
    'french': french_text
})

# Step 3: Basic Cleaning Function
def clean_sentence(sentence):
    sentence = sentence.lower()  # Lowercase the sentence
    sentence = re.sub(r"[^a-zA-Z0-9,.'’ ]+", "", sentence)  # Remove unwanted characters
    return sentence.strip()

# Apply the cleaning function
df['english'] = df['english'].apply(clean_sentence)
df['french'] = df['french'].apply(clean_sentence)

# Step 4: Split the Data
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Step 5: Save Preprocessed Data (Check file write operation)
train_output_file = '/home/dl/train_translations.csv'
test_output_file = '/home/dl/test_translations.csv'

train.to_csv(train_output_file, index=False)
test.to_csv(test_output_file, index=False)

print(f"Train data saved to: {train_output_file}")
print(f"Test data saved to: {test_output_file}")

# Step 6: Tokenization
# Initialize tokenizers for both English and French text
english_tokenizer = Tokenizer()
french_tokenizer = Tokenizer()

# Fit the tokenizers on the training data to create the vocabulary
english_tokenizer.fit_on_texts(train['english'])
french_tokenizer.fit_on_texts(train['french'])

# Convert sentences to sequences of integers for train data
train_english_seq = english_tokenizer.texts_to_sequences(train['english'])
train_french_seq = french_tokenizer.texts_to_sequences(train['french'])

# Pad sequences for uniform length for train data
max_len_english = max(len(seq) for seq in train_english_seq)
max_len_french = max(len(seq) for seq in train_french_seq)
train_english_padded = pad_sequences(train_english_seq, maxlen=max_len_english, padding='post')
train_french_padded = pad_sequences(train_french_seq, maxlen=max_len_french, padding='post')

# Convert sentences to sequences of integers for test data (separate from train data)
test_english_seq = english_tokenizer.texts_to_sequences(test['english'])
test_french_seq = french_tokenizer.texts_to_sequences(test['french'])

# Pad sequences for uniform length for test data
test_english_padded = pad_sequences(test_english_seq, maxlen=max_len_english, padding='post')
test_french_padded = pad_sequences(test_french_seq, maxlen=max_len_french, padding='post')

# Save the padded sequences as new columns in the train and test dataframes
train['english_padded'] = list(train_english_padded)
train['french_padded'] = list(train_french_padded)
test['english_padded'] = list(test_english_padded)
test['french_padded'] = list(test_french_padded)

# Save the final dataframes with padded sequences (Debugging file save)
train_padded_file = '/home/dl/train_translations_p.csv'
test_padded_file = '/home/dl/test_translations_p.csv'

train.to_csv(train_padded_file, index=False)
test.to_csv(test_padded_file, index=False)

print(f"Padded train data saved to: {train_padded_file}")
print(f"Padded test data saved to: {test_padded_file}")

print("Preprocessing complete. Ready for model training.")

