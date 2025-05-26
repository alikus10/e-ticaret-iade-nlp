import os
import warnings
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# --- Uyarıları bastır ---
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Ayarlar ---
input_text = "change mind"
model_dir = "output/word2vec_models"
lemmatized_df = pd.read_csv("output/lemmatized_sentences.csv")
stemmed_df = pd.read_csv("output/stemmed_sentences.csv")

# --- Cümle ortalama vektörü ---
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return None
    return np.mean(vectors, axis=0).reshape(1, -1)

# --- Benzerlik hesaplayıcı ---
def run_similarity(models_folder, input_text, text_df, prefix):
    model_files = [f for f in os.listdir(models_folder) if f.endswith(".model") and f.startswith(prefix)]

    for model_file in model_files:
        model_path = os.path.join(models_folder, model_file)
        model = Word2Vec.load(model_path)
        print(f"\n\U0001F4CC Model: {model_file}")

        input_vec = sentence_vector(input_text, model)
        if input_vec is None:
            print("Giriş metni vektörlenemedi.")
            continue

        similarities = []
        for idx, row in text_df.iterrows():
            sent = row.iloc[0]  # Uyar\u0131s\u0131z ve uyumlu
            vec = sentence_vector(sent, model)
            if vec is not None:
                sim = cosine_similarity(input_vec, vec)[0][0]
                similarities.append((sent, sim))

        # En benzer 5 sonucu al
        top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        for i, (sent, sim) in enumerate(top5, start=1):
            print(f"{i}. '{sent}' \u2192 Benzerlik: {sim:.4f}")

# --- LEMMATIZED modeller ---
run_similarity(model_dir, input_text, lemmatized_df, prefix="word2vec_lemmatized")

# --- STEMMED modeller ---
run_similarity(model_dir, input_text, stemmed_df, prefix="word2vec_stemmed")
