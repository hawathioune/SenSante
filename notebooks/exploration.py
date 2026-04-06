import pandas as pd

df = pd.read_csv("C:/Users/use/Documents/SenSante/data/patients_dakar.csv")

print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)
print(f"Nombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

print("--- 5 premiers patients ---")
print(df.head())

print("--- Statistiques descriptives ---")
print(df.describe().round(2))

print("--- Repartition des diagnostics ---")
for diag, count in df["diagnostic"].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {diag} : {count} patients ({pct:.1f}%)")

print("--- Repartition par region top 5 ---")
for region, count in df["region"].value_counts().head(5).items():
    print(f"  {region} : {count} patients")

print("--- Temperature moyenne par diagnostic ---")
for diag, temp in df.groupby("diagnostic")["temperature"].mean().items():
    print(f"  {diag} : {temp:.1f} C")