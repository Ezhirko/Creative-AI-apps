from flask import Flask, render_template, request, jsonify
import os
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
import torch
from torchtext.data.utils import get_tokenizer
import random

# Audio libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import soundfile as sf
import noisereduce as nr

app = Flask(__name__)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

from nltk.corpus import words
# Get a list of English words for inserting random words
english_words = words.words()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_text', methods=['POST'])
def upload_text():
    file = request.files['file']
    text_content = file.read().decode('utf-8')
    return jsonify({"text": text_content})

def preprocess_text(text, operation):
    if operation == "Tokenization":
        tokenizer = get_tokenizer('basic_english')
        tokens = tokenizer(text)
        # Convert tokens to a tensor and pad/truncate them to max_length
        token_ids = torch.tensor([ord(token[0]) for token in tokens], dtype=torch.long)  # Simple token encoding for example
        return token_ids.tolist()
    elif operation == "convert to lower case":
        return text.lower()
    elif operation == "Remove punctuation":
        return re.sub(r'[^\w\s]', '', text)
    elif operation == "Remove Stopwords":
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in tokens if word.lower() not in stop_words])
    elif operation == "Stemming":
        stemmer = PorterStemmer()
        tokens = word_tokenize(text)
        return ' '.join([stemmer.stem(word) for word in tokens])
    elif operation == "Lemmatization":
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        return ' '.join([lemmatizer.lemmatize(word) for word in tokens])
    elif operation == "Remove Emojis":
        return re.sub(r'[^\x00-\x7F]+', '', text)
    else:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in tokens if word.lower() not in stop_words])
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        text = ' '.join([lemmatizer.lemmatize(word) for word in tokens])
        text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

@app.route('/preprocess', methods=['POST'])
def preprocess():
    text = request.json['text']
    operation = request.json['operation']
    result = preprocess_text(text, operation)
    return jsonify({"result": result})

def get_synonym(word):
    """Get a synonym for a given word, if available, otherwise return the original word."""
    synonyms = wordnet.synsets(word)
    # Filter synonyms by the same part of speech as the original word
    filtered_synonyms = [lemma.name() for syn in synonyms for lemma in syn.lemmas() if lemma.name().lower() != word.lower()]
    
    if filtered_synonyms:
        return random.choice(filtered_synonyms)  # Pick a random synonym
    return word  # Return the original word if no synonym is found

def swap_words_randomly(text):
    # Tokenize the text into words and punctuation
    words = nltk.word_tokenize(text)
    
    # Separate words and punctuation marks
    words_only = [word for word in words if word.isalnum()]  # Only alphabetic or numeric tokens
    punctuation = [(i, word) for i, word in enumerate(words) if not word.isalnum()]  # Punctuation with indices

    # Shuffle the words randomly
    random.shuffle(words_only)
    
    # Reassemble the text, keeping punctuation in place
    result = words_only
    for index, punct in punctuation:
        result.insert(index, punct)
        
    # Join the tokens back into a single string
    return ' '.join(result)

def random_insert_or_delete(text, operation="insert"):
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    
    if operation == "insert":
        # Choose a random word to insert and a random position
        random_word = random.choice(english_words)
        position = random.randint(0, len(words))
        words.insert(position, random_word)
        print(f"Inserted '{random_word}' at position {position}")
    
    elif operation == "delete" and len(words) > 1:
        # Choose a random word to delete
        position = random.randint(0, len(words) - 1)
        removed_word = words.pop(position)
        print(f"Deleted '{removed_word}' from position {position}")
    
    # Join the tokens back into a single string
    return ' '.join(words)

# Dummy augmentation function for simplicity
def augment_text(text, operation):
    if operation == "Synonyms Replacement":
        words = word_tokenize(text)
        new_text = [get_synonym(word) if wordnet.synsets(word) else word for word in words]
        return ' '.join(new_text)
    elif operation == "Random Swapping":
        return swap_words_randomly(text)
    elif operation == "Random Insertion/Deletion":
        return random_insert_or_delete(text)
    return text

@app.route('/augment', methods=['POST'])
def augment():
    text = request.json['text']
    operation = request.json['operation']
    result = augment_text(text, operation)
    return jsonify({"result": result})

def generate_oscillogram(audio_data, sample_rate):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Oscillogram")
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    oscillogram_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    return oscillogram_base64

def generate_mfcc_spectrogram(audio_data, sample_rate):
    # Compute the MFCCs from the audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCC Spectrogram")
    plt.tight_layout()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    mfcc_spectrogram_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    return mfcc_spectrogram_base64

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio_file']
    audio_data, sample_rate = sf.read(io.BytesIO(audio_file.read()))

    oscillogram = generate_oscillogram(audio_data, sample_rate)
    mfcc_spectrogram = generate_mfcc_spectrogram(audio_data, sample_rate)

    return jsonify({
        "oscillogram": oscillogram,
        "spectrogram": mfcc_spectrogram
    })

@app.route('/preprocess_audio', methods=['POST'])
def preprocess_audio():
    print('Entered preprocess_audio')
    audio_file = request.files['file']
    operation = request.form['operation']
    
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    if operation == "Normalization":
        # Normalize audio
        max_amplitude = np.max(np.abs(y))
        if max_amplitude > 0:
            y_normalized = y / max_amplitude
        else:
            y_normalized = y
        
        output_file_path = "static/processed_normalized.wav"
        sf.write(output_file_path, y_normalized, sr)
        oscillogram = generate_oscillogram(y_normalized, sr)
        mfcc_spectrogram = generate_mfcc_spectrogram(y_normalized, sr)

    elif operation == "Noise Reduction":
        # Implement noise reduction (you can use a library like noisereduce)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        output_file_path = "static/processed_denoised.wav"
        sf.write(output_file_path, y_denoised, sr)
        oscillogram = generate_oscillogram(y_denoised, sr)
        mfcc_spectrogram = generate_mfcc_spectrogram(y_denoised, sr)

    elif operation == "Resampling":
        # Resample the audio to 16kHz
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
        output_file_path = "static/processed_resampled.wav"
        sf.write(output_file_path, y_resampled, 16000)
        oscillogram = generate_oscillogram(y_resampled, sr)
        mfcc_spectrogram = generate_mfcc_spectrogram(y_resampled, sr)

    return jsonify({"processed_file": output_file_path,
                    "oscillogram": oscillogram,
                    "spectrogram": mfcc_spectrogram})

@app.route('/augment_audio', methods=['POST'])
def augment_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    audio_file = request.files['file']
    augment_option = request.form['option']

    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    if augment_option == "Random Noise":
        noise = np.random.randn(len(y))
        augmented_audio = y + 0.005 * noise
    elif augment_option == "Time Stretching":
        augmented_audio = librosa.effects.time_stretch(y, rate=1.5)  # Speed up by 50%
    elif augment_option == "Pitch Shifting":
        augmented_audio = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)  # Shift pitch up by 4 semitones
    elif augment_option == "Change Speed":
        augmented_audio = librosa.effects.time_stretch(y, rate=0.8)  # Slow down by 20%
    else:
        return jsonify({'error': 'Invalid augmentation option'}), 400

    # Save the augmented audio to a file
    augmented_file_path = os.path.join('static', 'augmented_audio.wav')
    sf.write(augmented_file_path, augmented_audio, sr)
    oscillogram = generate_oscillogram(augmented_audio, sr)
    mfcc_spectrogram = generate_mfcc_spectrogram(augmented_audio, sr)

    return jsonify({'augmented_file': augmented_file_path,
                    "oscillogram": oscillogram,
                    "spectrogram": mfcc_spectrogram})

if __name__ == '__main__':
    app.run(debug=True)
