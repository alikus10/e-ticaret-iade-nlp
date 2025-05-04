import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# 1. Stemmed dosyasını yükle
df = pd.read_csv("output/stemmed_sentences.csv")

# 2. Tüm metinleri birleştir
all_text = " ".join(df["stemmed_text"].astype(str).tolist())

# 3. Kelimelere ayır
tokens = all_text.split()

# 4. Kelime frekanslarını say
word_freq = Counter(tokens)
most_common = word_freq.most_common(1000)

# 5. Sıralama ve frekansları çıkar
ranks = np.arange(1, len(most_common) + 1)
frequencies = np.array([freq for _, freq in most_common])

# 6. Grafiği çiz
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker=".")
plt.title("Zipf Yasası - Stemmed Veri")
plt.xlabel("Kelime Sıklığı Sırası (log)")
plt.ylabel("Frekans (log)")
plt.grid(True)
plt.tight_layout()
plt.show()
