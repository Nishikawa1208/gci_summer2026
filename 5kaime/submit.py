import pandas as pd
import numpy as np


def homework(target_online_retail_data_tb, n):
    """
    顧客ごとの購入金額合計を出し、金額の大きい順に並べて上からn等分した
    各グループの売上比率を、売上の大きい順に並べたSeriesを返します。

    引数:
      target_online_retail_data_tb: 前処理済みのDataFrame（'TotalPrice'列を含む）
      n: 分割数（自然数）

    戻り値:
      Pandas.Series（インデックスが「グループ1」「グループ2」...、比率の合計は1）
    """
    # CustomerIDごとに購入金額を合計する
    totals = target_online_retail_data_tb.groupby('CustomerID', dropna=True)['TotalPrice'].sum()

    # 合計金額で降順に並べる
    totals_sorted = totals.sort_values(ascending=False)

    # 上位からn等分するため、行番号の配列を作る
    idx = np.arange(len(totals_sorted))

    # pd.qcutで等分（labels=Falseで0..n-1のグループ番号になる）
    groups = pd.qcut(idx, q=n, labels=False, duplicates='drop')

    # グループごとの売上合計を計算
    group_sum = totals_sorted.groupby(groups).sum()

    # 全体に対する比率を求める
    total_sales = group_sum.sum()
    if total_sales == 0:
        shares = group_sum.astype(float)
    else:
        shares = group_sum / total_sales

    # 大きい順に並べ替える
    shares_sorted = shares.sort_values(ascending=False)

    # インデックスを「グループ1」形式にする
    labels = [f"グループ{i+1}" for i in range(len(shares_sorted))]
    shares_sorted.index = labels

    return pd.Series(shares_sorted)


if __name__ == '__main__':
    # 簡易テスト（workspace にある student-mat.csv 等は使用せず、関数呼び出し例のみ）
    print('submit.py: homework 関数を定義しました。')
