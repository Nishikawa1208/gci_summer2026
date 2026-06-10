def homework(path_winequality_data, X_column, Y_column):
    # CSVを直接読み込み（セミコロン区切り）して単回帰の決定係数を計算する
    with open(path_winequality_data, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split(';')
        # ヘッダのクォートを剥く
        header = [h.strip().strip('"') for h in header]
        try:
            ix = header.index(X_column)
            iy = header.index(Y_column)
        except ValueError:
            raise ValueError('指定したカラム名が見つかりません')

        xs = []
        ys = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            # 値が不足している行は無視
            if len(parts) <= max(ix, iy):
                continue
            try:
                xv = float(parts[ix].strip().strip('"'))
                yv = float(parts[iy].strip().strip('"'))
            except Exception:
                # 数値変換できない行は無視
                continue
            xs.append(xv)
            ys.append(yv)

    if not xs or not ys:
        raise ValueError('有効なデータがありません')

    n = len(xs)
    xm = sum(xs) / n
    ym = sum(ys) / n

    num = 0.0
    den = 0.0
    for xi, yi in zip(xs, ys):
        dx = xi - xm
        num += dx * (yi - ym)
        den += dx * dx

    if den == 0.0:
        # 説明変数に分散が無ければ決定係数は定義されない
        return 0.0

    slope = num / den
    intercept = ym - slope * xm

    ss_res = 0.0
    ss_tot = 0.0
    for xi, yi in zip(xs, ys):
        yhat = slope * xi + intercept
        ss_res += (yi - yhat) ** 2
        ss_tot += (yi - ym) ** 2

    if ss_tot == 0.0:
        return 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)
