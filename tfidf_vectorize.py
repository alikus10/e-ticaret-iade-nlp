import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatized_df = pd.read_csv("output/lemmatized_sentences.csv")
stemmed_df = pd.read_csv("output/stemmed_sentences.csv")


lemmatized_texts = lemmatized_df["lemmatized_text"].astype(str).tolist()
stemmed_texts = stemmed_df["stemmed_text"].astype(str).tolist()


vectorizer = TfidfVectorizer()


tfidf_lemmatized = vectorizer.fit_transform(lemmatized_texts)
tfidf_lemmatized_df = pd.DataFrame(tfidf_lemmatized.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_lemmatized_df.to_csv("output/tfidf_lemmatized.csv", index=False)
print("✅ tfidf_lemmatized.csv kaydedildi.")

vectorizer = TfidfVectorizer()
tfidf_stemmed = vectorizer.fit_transform(stemmed_texts)
tfidf_stemmed_df = pd.DataFrame(tfidf_stemmed.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_stemmed_df.to_csv("output/tfidf_stemmed.csv", index=False)
print("✅ tfidf_stemmed.csv kaydedildi.")
