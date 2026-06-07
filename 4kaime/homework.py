import numpy as np
import pandas as pd
from pandas import DataFrame

url_winequality_data = "/content/drive/MyDrive/Colab Notebooks/GCI 2024 Winter/第4回_Pythonによるデータ加工処理の基礎（Pandas）_宿題あり/winequality-red.csv"

def homework(url_winequality_data, n):
    # CSV を読み込み（winequality データは ';' 区切り）
    df = pd.read_csv(url_winequality_data, sep=';')

    # 引数チェック
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if n > len(df):
        raise ValueError("n must not exceed number of rows in the data")

    va_cat = pd.qcut(df['volatile acidity'], q=n)
    df = df.copy()
    df['va_group'] = va_cat

    quality5 = df[df['quality'] == 5]
    means = quality5.groupby('va_group')['alcohol'].mean()

    if means.empty:
        my_result = float('nan')
    else:
        my_result = float(means.min())

    return my_result
