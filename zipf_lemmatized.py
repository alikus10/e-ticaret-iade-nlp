import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("output/lemmatized_sentences.csv")

all_text = " ".join(df["lemmatized_text"].astype(str).tolist())

tokens = all_text.split()


word_freq = Counter(tokens)
most_common = word_freq.most_common(1000)


ranks = np.arange(1, len(most_common) + 1)
frequencies = np.array([freq for _, freq in most_common])


plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker=".")
plt.title("Zipf Yasası - Lemmatized Veri")
plt.xlabel("Kelime Sıklığı Sırası (log)")
plt.ylabel("Frekans (log)")
plt.grid(True)
plt.tight_layout()
plt.show()
