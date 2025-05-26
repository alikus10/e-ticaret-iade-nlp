import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import re

print("📥 Veri yükleniyor...")
df = pd.read_csv("data/ecommerce_returns_synthetic_data.csv")

df = df.dropna(subset=["Return_Reason"])
print(f"🧾 {len(df)} adet geçerli iade gerekçesi bulundu.")

df["clean_text"] = df["Return_Reason"].str.lower()
df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

all_text = " ".join(df["clean_text"])
tokens = all_text.split()

word_freq = Counter(tokens)
most_common = word_freq.most_common(1000)

ranks = np.arange(1, len(most_common) + 1)
frequencies = np.array([freq for _, freq in most_common])

plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker=".")
plt.title("Zipf Yasası - Ham Veri (Return_Reason)")
plt.xlabel("Kelime Sıklığı Sırası (log)")
plt.ylabel("Frekans (log)")
plt.grid(True)
plt.tight_layout()
plt.show()
