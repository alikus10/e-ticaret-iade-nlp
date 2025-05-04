
# 🛍️ E-Ticaret İade Gerekçesi Benzerlik Sınıflandırması

Bu proje, e-ticaret platformlarındaki iade nedenlerini doğal dil işleme (NLP) teknikleri ile analiz ederek benzer gerekçeleri sınıflandırmayı amaçlar. Proje, `ecommerce_returns_synthetic_data.csv` adlı sentetik veri seti üzerinde yürütülmüştür.

## 🔍 Amaç
- İade gerekçelerini metinsel olarak analiz etmek
- Temizlenmiş veriler üzerinde TF-IDF ve Word2Vec yöntemlerini uygulamak
- Anlamlı kelime benzerlikleri ve sınıflandırma potansiyeli ortaya koymak

## 📁 Klasör Yapısı

```
ecommerce-returns-nlp/
├── data/
│   └── ecommerce_returns_synthetic_data.csv
├── output/
│   ├── lemmatized_sentences.csv
│   ├── stemmed_sentences.csv
│   ├── tfidf_lemmatized.csv
│   ├── tfidf_stemmed.csv
│   └── word2vec_models/
│       ├── word2vec_lemmatized_cbow_win2_dim100.model
│       ├── ...
├── preprocessing.py
├── tfidf_vectorize.py
├── word2vec_train.py
├── word2vec_model_test.py
└── README.md
```

## 🧼 Ön İşleme Aşamaları

- Küçük harfe çevirme
- Noktalama ve özel karakter temizliği
- Stopword çıkarımı (NLTK)
- Lemmatization (spaCy)
- Stemming (PorterStemmer)

Çıktılar:
- `lemmatized_sentences.csv`
- `stemmed_sentences.csv`

## 📊 Zipf Yasası Analizi

Her veri seti için log-log Zipf grafiği çizilmiştir:
- `zipf_raw.py` → Ham veri
- `zipf_lemmatized.py` → Lemmatized veri
- `zipf_stemmed.py` → Stemmed veri

Grafikler `matplotlib` kullanılarak görselleştirilmiştir.

## 📈 TF-IDF Vektörleştirme

- Uygulanan dosyalar: `tfidf_vectorize.py`
- Kullanılan araç: `TfidfVectorizer` (scikit-learn)
- Çıktılar:
  - `tfidf_lemmatized.csv`
  - `tfidf_stemmed.csv`

## 🧠 Word2Vec Model Eğitimi

Gensim kullanılarak toplam **16 Word2Vec modeli** eğitilmiştir.  
Eğitim parametreleri:

```python
parameters = [
  {'model_type': 'cbow', 'window': 2, 'vector_size': 100},
  {'model_type': 'skipgram', 'window': 2, 'vector_size': 100},
  ...
  {'model_type': 'skipgram', 'window': 4, 'vector_size': 300}
]
```

Model isimlendirme:
```
word2vec_lemmatized_cbow_win2_dim100.model
word2vec_stemmed_skipgram_win4_dim300.model
...
```

Her model için `"wrong"` gibi anahtar kelimelerle benzer kelimeler analiz edilmiştir.

## 🧪 Örnek Test: Word2Vec

```python
from gensim.models import Word2Vec
model = Word2Vec.load("output/word2vec_models/word2vec_lemmatized_cbow_win2_dim100.model")
print(model.wv.most_similar("wrong", topn=5))
```

Örnek çıktı:
```
[('item', 0.91), ('product', 0.89), ('received', 0.87), ('different', 0.85), ('sent', 0.84)]
```

## ⚙️ Gereksinimler

```bash
pip install pandas nltk spacy gensim scikit-learn matplotlib
python -m nltk.downloader stopwords
python -m spacy download en_core_web_sm
```

## 📌 Lisans

Bu proje eğitim amaçlıdır.
