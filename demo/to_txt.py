import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("ais_all_sorted.csv")

# 1️⃣ Olduğu gibi (virgüller korunarak) TXT olarak kaydet
#df.to_csv("ais_all_sorted.txt", index=False)

# 2️⃣ İstersen tab ile ayrılmış olarak kaydet
#df.to_csv("ais_all_sorted_tab.txt", index=False, sep="\t")

# 3️⃣ İstersen boşluk ile ayrılmış olarak kaydet
df.to_csv("ais_all_sorted_space.txt", index=False, sep=" ")
