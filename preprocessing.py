import pandas as pd
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Gerekli kaynakları indir (sadece ilk çalıştırmada gerekir)
nltk.download('stopwords')
spacy.cli.download("en_core_web_sm")

# SpaCy modeli yükle
nlp = spacy.load("en_core_web_sm")

print("🔄 CSV yükleniyor...")

# CSV dosyasını oku
try:
    df = pd.read_csv("data/ecommerce_returns_synthetic_data.csv")
    print("✅ CSV başarıyla yüklendi.")
except Exception as e:
    print("❌ CSV yüklenemedi:", e)
    exit()

# NaN olan gerekçeleri çıkar
df = df.dropna(subset=["Return_Reason"]).reset_index(drop=True)
print(f"💡 {len(df)} adet geçerli iade gerekçesi bulundu.")

# Küçük harfe çevir ve özel karakterleri sil
df["clean_text"] = df["Return_Reason"].str.lower()
df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# Tokenization + stopword çıkarma
stop_words = set(stopwords.words('english'))
df["tokens"] = df["clean_text"].apply(lambda x: [word for word in x.split() if word not in stop_words])

# Lemmatization fonksiyonu
def lemmatize(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

# Stemming fonksiyonu
stemmer = PorterStemmer()
def stem(tokens):
    return [stemmer.stem(word) for word in tokens]

print("🔧 Lemmatization ve stemming uygulanıyor...")

df["lemmatized"] = df["tokens"].apply(lemmatize)
df["stemmed"] = df["tokens"].apply(stem)

# Tekrar metne dönüştür
df["lemmatized_text"] = df["lemmatized"].apply(lambda x: " ".join(x))
df["stemmed_text"] = df["stemmed"].apply(lambda x: " ".join(x))

# CSV olarak kaydet
df[["lemmatized_text"]].to_csv("output/lemmatized_sentences.csv", index=False)
df[["stemmed_text"]].to_csv("output/stemmed_sentences.csv", index=False)

print("✅ Temizlik tamamlandı.")
print("📁 Kaydedilen dosyalar:")
print("→ output/lemmatized_sentences.csv")
print("→ output/stemmed_sentences.csv")
