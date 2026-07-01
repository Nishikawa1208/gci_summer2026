import pandas as pd

# データの読み込み
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# データのサイズ（行数・列数）を確認
print(f"Trainデータのサイズ: {train.shape}")
print(f"Testデータのサイズ: {test.shape}")

# 修正箇所：display(...) ではなく print(...) を使います
print("\n--- Trainデータの先頭5行 ---")
print(train.head())

# データの型や欠損値の状況を確認
print("\n--- Trainデータの基本情報 ---")
train.info()

print("\n--- Testデータの基本情報 ---")
test.info()
