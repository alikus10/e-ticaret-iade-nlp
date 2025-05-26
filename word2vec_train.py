import pandas as pd
import os
from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences


output_dir = "output/word2vec_models"
os.makedirs(output_dir, exist_ok=True)


parameters = [
    {'model_type': 'cbow',     'sg': 0, 'window': 2, 'vector_size': 100},
    {'model_type': 'skipgram', 'sg': 1, 'window': 2, 'vector_size': 100},
    {'model_type': 'cbow',     'sg': 0, 'window': 4, 'vector_size': 100},
    {'model_type': 'skipgram', 'sg': 1, 'window': 4, 'vector_size': 100},
    {'model_type': 'cbow',     'sg': 0, 'window': 2, 'vector_size': 300},
    {'model_type': 'skipgram', 'sg': 1, 'window': 2, 'vector_size': 300},
    {'model_type': 'cbow',     'sg': 0, 'window': 4, 'vector_size': 300},
    {'model_type': 'skipgram', 'sg': 1, 'window': 4, 'vector_size': 300},
]

lemmatized_df = pd.read_csv("output/lemmatized_sentences.csv")
stemmed_df = pd.read_csv("output/stemmed_sentences.csv")


lemmatized_sentences = [row.split() for row in lemmatized_df["lemmatized_text"].dropna()]
stemmed_sentences = [row.split() for row in stemmed_df["stemmed_text"].dropna()]

for data_type, sentences in [("lemmatized", lemmatized_sentences), ("stemmed", stemmed_sentences)]:
    for p in parameters:
        print(f"🎯 Eğitim başlatılıyor: {data_type} - {p['model_type']} - win={p['window']} - dim={p['vector_size']}")

        model = Word2Vec(
            sentences,
            vector_size=p['vector_size'],
            window=p['window'],
            sg=p['sg'],
            min_count=1,
            workers=4,
            epochs=20
        )

        model_filename = f"word2vec_{data_type}_{p['model_type']}_win{p['window']}_dim{p['vector_size']}.model"
        model_path = os.path.join(output_dir, model_filename)
        model.save(model_path)

        print(f"✅ Model kaydedildi: {model_filename}")
