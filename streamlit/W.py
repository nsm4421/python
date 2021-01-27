
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(years = 10, seed = 1221):
    # 샘플 (월별)데이터 만들기
    np.random.seed(seed)
    years = years
    maintain_count = np.sort(np.random.randint(low = 100000, high = 300000, size = years*12))
    termination_count = np.random.randint(low = 2000, high = 5000, size = years*12)
    df_month = pd.DataFrame({'유지계약건수' : maintain_count, '해지건수' : termination_count}, index = range(1, years*12+1))
    df_month.index = df_month.index.set_names(['경과월수'])
    df_month.head()

    # 월별 유지율, 해지율
    df_month['월별해지율'] = df_month['해지건수'] / df_month['유지계약건수']
    df_month['월별유지율'] = 1 - df_month['월별해지율']
    df_month.head()

    # 연도별 유지율, 해지율
    maintain_rate_annual = [np.product(df_month['월별유지율'][12*year:12*(year+1)]) for year in range(years)]
    terminate_rate = [1 - rate for rate in maintain_rate_annual]
    df_annual = pd.DataFrame({'유지율' : maintain_rate_annual, '해지율' : terminate_rate}, index = range(1, years+1))
    df_annual.index = df_annual.index.set_names(['경과년수'])
    df_annual.head()

    # Skew 구하기
    skew = []
    for month in range(years*12):
        w_s = df_month['월별해지율'].values[month]
        w_t = df_annual['해지율'].values[month//12]
        sk_s = np.log(1-w_s)/np.log(1-w_t)
        skew.append(sk_s)
    df_month['skew'] = skew
    df_month['skew해지율'] = [1-(1-df_annual['해지율'].values[month//12])**skew[month] for month in range(12*years)]

    return df_month, df_annual