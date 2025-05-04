import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import re

# 1. Ham veriyi yükle
print("📥 Veri yükleniyor...")
df = pd.read_csv("data/ecommerce_returns_synthetic_data.csv")

# 2. Boş gerekçeleri çıkar
df = df.dropna(subset=["Return_Reason"])
print(f"🧾 {len(df)} adet geçerli iade gerekçesi bulundu.")

# 3. Küçük harfe çevir + noktalama temizliği
df["clean_text"] = df["Return_Reason"].str.lower()
df["clean_text"] = df["clean_text"].apply(lambda x: re.sub(r'[^a-z\s]', '', x))

# 4. Tüm metinleri birleştir ve tokenlara ayır
all_text = " ".join(df["clean_text"])
tokens = all_text.split()

# 5. Kelime frekanslarını hesapla
word_freq = Counter(tokens)
most_common = word_freq.most_common(1000)

# 6. Sıralama ve frekans listesi
ranks = np.arange(1, len(most_common) + 1)
frequencies = np.array([freq for _, freq in most_common])

# 7. Zipf log-log grafiğini çiz
plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies, marker=".")
plt.title("Zipf Yasası - Ham Veri (Return_Reason)")
plt.xlabel("Kelime Sıklığı Sırası (log)")
plt.ylabel("Frekans (log)")
plt.grid(True)
plt.tight_layout()
plt.show()
