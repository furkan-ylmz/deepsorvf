import pandas as pd
from glob import glob

# Tüm CSV dosyalarını oku
files = glob("ais/*.csv")
df_list = [pd.read_csv(f) for f in files]

# Birleştir
df_all = pd.concat(df_list, ignore_index=True)

# timestamp’e göre sırala
df_all = df_all.sort_values(by="timestamp").reset_index(drop=True)

# Tek dosya olarak kaydet
df_all.to_csv("ais_all_sorted.csv", index=False)