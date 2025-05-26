import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Adım: CSV dosyalarını oku (output klasöründen) ---
tfidf_lem = pd.read_csv("output/tfidf_lemmatized.csv")
texts = pd.read_csv("output/lemmatized_sentences.csv")["lemmatized_text"]

# --- 2. Adım: Giriş metnini belirle ---
query_text = "change mind"

# Giriş metninin index'ini bul
query_indices = texts[texts == query_text].index

if len(query_indices) == 0:
    print("Giriş metni veri setinde bulunamadı.")
else:
    query_index = query_indices[0]
    query_vec = tfidf_lem.iloc[query_index].values.reshape(1, -1)

    # --- 3. Adım: Tüm benzerlikleri hesapla ---
    similarities = cosine_similarity(query_vec, tfidf_lem.values).flatten()

    # Kendisi hariç, en benzer 5 sonucu al
    similar_indices = similarities.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != query_index][:5]

    print("🔍 Giriş metni:", query_text)
    print("\n📌 En benzer 5 metin:")
    for i, idx in enumerate(similar_indices, start=1):
        print(f"{i}. '{texts.iloc[idx]}' → Benzerlik: {similarities[idx]:.4f}")
