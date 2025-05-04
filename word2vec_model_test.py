from gensim.models import Word2Vec

# Eğitilmiş modeli yükle
model_path = "output/word2vec_models/word2vec_lemmatized_cbow_win2_dim100.model"
model = Word2Vec.load(model_path)

# "wrong" kelimesine en yakın 5 kelimeyi yazdır
similar_words = model.wv.most_similar("wrong", topn=5)

# Ekrana bastır
print("Benzer kelimeler (word2vec_lemmatized_cbow_win2_dim100.model):")
for word, score in similar_words:
    print(f"{word} → {score:.2f}")
