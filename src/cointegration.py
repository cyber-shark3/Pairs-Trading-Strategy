from statsmodels.tsa.stattools import coint


def test_cointegration(data, asset1, asset2):
    score, pvalue, _ = coint(data[asset1], data[asset2])
    print(
        f"Cointegration p-value for {asset1} and {asset2}: "
        f"{pvalue:.4f}"
    )
    return pvalue < 0.05
