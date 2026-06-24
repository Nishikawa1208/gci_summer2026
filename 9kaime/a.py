import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def homework(path_winequality_data, n):

    df = pd.read_csv(path_winequality_data, sep=';')

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(scaled_data)

    cluster_counts = np.bincount(kmeans.labels_)

    # np.array[int]型で返す
    return cluster_counts.astype(int)

# --- a.py の一番下のテスト用コードを書き換え ---

if __name__ == "__main__":
    # フルパス（絶対パス）でファイルのある場所を直接指定します
    file_path = winequality-red.csv

    # 関数の実行（クラスター数は3）
    result = homework(file_path, 3)

    # 結果を表示
    print("--- 実行結果 ---")
    print(result)
