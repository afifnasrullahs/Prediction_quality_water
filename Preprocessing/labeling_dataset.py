import pandas as pd
import os


out_dir = r"C:\Users\afifn\Documents\TES MULTIMEDIA SOLUSI PRIMA\Dataset\processed"
os.makedirs(out_dir, exist_ok=True)


def get_label(row):
    ph = row['ph']
    turb = row['turbidity']
    tds  = row['tds']

    if (ph < 5.0 or ph > 9.5) or (turb > 30) or (tds > 1500):
        return"Coklat"
    elif (5.0 <= ph <= 6.0 or 8.5 <= ph <= 9.5) or (15 <= turb <= 30) or (800 <= tds <= 1500):
        return "Orange"
    elif (6.0 <= ph <= 6.5 or 8.0 <= ph <= 8.5) or (5 <= turb <= 15) or (500 <= tds <= 800):
        return "Biru"
    elif (6.5 <= ph <= 8.0) and (turb < 5) and (tds < 500):
        return "putih"
    else:
        return "tidak ada"

# Labeling dataset A
file1 = r"C:\Users\afifn\Documents\TES MULTIMEDIA SOLUSI PRIMA\Dataset\raw\_PanelAs__202505281600 new.csv"
df1 = pd.read_csv(file1)

df1["quality_label"] = df1.apply(get_label, axis=1)

output1 = os.path.join(out_dir, "panelAs_labeled.csv")
df1.to_csv(output1, index = False)

print("dataset 1 selesai :", output1)

# Labeling dataset B
file2 = r"C:\Users\afifn\Documents\TES MULTIMEDIA SOLUSI PRIMA\Dataset\raw\_PanelBs__202506041149 new.csv"
df2 = pd.read_csv(file2)

df2["quality_label"] = df2.apply(get_label, axis=1)

output2 = os.path.join(out_dir, "panelBs_labeled.csv")
df2.to_csv(output2, index = False)

print("dataset 2 selesai :", output2)