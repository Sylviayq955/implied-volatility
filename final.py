import pandas as pd
import tushare as ts
import time
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

ts.set_token('99dca971a24a131665c46e03281918a16db309ca7c7863f3658d9762')
ts_pro = ts.pro_api()


# 获取进行选股的股票池
def get_SZ50_stocks(start, end):
    # 获取上证50成分股
    df1 = ts_pro.index_weight(index_code="000016.SH", start_date=start, end_date=end)
    SZ50_codes = df1["con_code"].tolist()
    # 剔除最近一年上市和st股票
    df2 = ts_pro.stock_basic(exchange="", list_status="L")
    df2 = df2[df2["list_date"].apply(int).values < 20200101]
    df2 = df2[-df2["name"].apply(lambda x:x.startswith("*ST"))]
    all_codes = df2["ts_code"].tolist()
    stocks_codes = []
    for i in SZ50_codes:
        if i in all_codes:
            stocks_codes.append(i)
    return stocks_codes

# 将股票分为六个组
def group_stocks(stocks, date):
    # 划分大小市值
    list_mv = []
    df_stocks = pd.DataFrame()
    count = 0
    for i in stocks:
        count += 1
        a = ts_pro.daily_basic(ts_code=i, trade_date=date)
        a = a["circ_mv"]
        a = a.tolist()
        list_mv += a
        time.sleep(0.3)
        print("第%d支股票市值计算完成" % count)
    df_stocks["code"] = stocks_codes
    df_stocks["mv"] = list_mv
    df_stocks["SB"] = df_stocks["mv"].map(lambda x: "B" if x > df_stocks["mv"].median() else "S")

    # 划分高中低账面市值比
    list_bm = []
    count = 0
    for i in stocks_codes:
        count += 1
        b = ts_pro.daily_basic(ts_code=i, trade_date=date)
        b = 1 / b["pb"]
        b = b.tolist()
        time.sleep(0.3)
        list_bm += b
        print("第%d支股票账面市值比计算完成" % count)
    df_stocks["bm"] = list_bm
    df_stocks["HML"] = df_stocks["bm"].apply(lambda x: "H" if x >= df_stocks["bm"].quantile(0.7)
    else ("L" if x <= df_stocks["bm"].quantile(0.3) else "M"))
    return df_stocks

# 计算日收益率
def groups_return(stocks, start, end):
    SL = stocks[stocks["SB"].isin(["S"]) & stocks["HML"].isin(["L"])].code.tolist()
    sum_SL = df_stocks[df_stocks["SB"].isin(["S"]) & df_stocks["HML"].isin(["L"])]["mv"].sum()
    SM = stocks[stocks["SB"].isin(["S"]) & stocks["HML"].isin(["M"])].code.tolist()
    sum_SM = df_stocks[df_stocks["SB"].isin(["S"]) & df_stocks["HML"].isin(["M"])]["mv"].sum()
    SH = stocks[stocks["SB"].isin(["S"]) & stocks["HML"].isin(["H"])].code.tolist()
    sum_SH = df_stocks[df_stocks["SB"].isin(["S"]) & df_stocks["HML"].isin(["H"])]["mv"].sum()
    BL = stocks[stocks["SB"].isin(["B"]) & stocks["HML"].isin(["L"])].code.tolist()
    sum_BL = df_stocks[df_stocks["SB"].isin(["B"]) & df_stocks["HML"].isin(["L"])]["mv"].sum()
    BM = stocks[stocks["SB"].isin(["B"]) & stocks["HML"].isin(["M"])].code.tolist()
    sum_BM = df_stocks[df_stocks["SB"].isin(["B"]) & df_stocks["HML"].isin(["M"])]["mv"].sum()
    BH = stocks[stocks["SB"].isin(["B"]) & stocks["HML"].isin(["H"])].code.tolist()
    sum_BH = df_stocks[df_stocks["SB"].isin(["B"]) & df_stocks["HML"].isin(["H"])]["mv"].sum()
    groups = [SL, SM, SH, BL, BM, BH]
    sums = [sum_SL, sum_SM, sum_SH, sum_BL, sum_BM, sum_BH]
    groups_names = ["SL", "SM", "SH", "BL", "BM", "BH"]
    df_groups = pd.DataFrame(columns=groups_names)
    count = 0
    for group in groups:
        df1 = pd.DataFrame()


        for i in range(len(group)):
            data = ts_pro.daily(ts_code=group[i], start_date=start, end_date=end)
            data.sort_values(by="trade_date", inplace=True)
            data = data["pct_chg"]*df_stocks["mv"][i]
            df1[group[i]] = data
        df_groups[groups_names[count]] = df1.apply(lambda x: x.sum()/sums[count], axis=1)/100
        print("%s组计算完成" % groups_names[count])
        count += 1

    return df_groups


# 计算每日SMB，HML
def SMB_HML(data):
    data["SMB"] = (data["SL"] + data["SM"] + data["SH"])/3 - (data["BL"] + data["BM"] + data["BH"])/3
    data["HML"] = (data["SH"] + data["BH"])/2 - (data["SL"] + data["BL"])/2

    return data

#加入市场因子和股票收益率
def selection(data, start, end, stocks_codes):
    MKT = ts_pro.index_daily(ts_code="000016.SH", start_date=start, end_date=end)
    MKT.sort_values(by="trade_date", ascending=True, inplace=True)
    MKT = (MKT["pct_chg"]/100-0.035).tolist()                       #tolist函数的作用：将数组或矩阵转化为列表
    data["MKT"] = MKT
    data.drop(data.columns[0:6], axis=1, inplace=True)
    count = 0
    for i in range(len(stocks_codes)):
        a = ts_pro.daily(ts_code=stocks_codes[i], start_date=20210101, end_date=20220101)
        if len(a) == 243:
            count += 1
            print(count)
            a.sort_values(by="trade_date", ascending=True, inplace=True)
            a = (a["pct_chg"]/100-0.035).tolist()
            data["%s"%stocks_codes[i]] = a

    return data

#回归计算阿尔法
def OLS(df_final):
    results = pd.DataFrame()
    stocks_return = df_final.iloc[:, 3:50]
    for i in range(len(stocks_return.columns)):
        x = df_final.iloc[:, 0:3]
        y = stocks_return.iloc[:, i]
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        result = model.fit()
        results[i] = result.params
    results.columns = stocks_return.columns
    results.rename(index={"const":"Alpha"}, inplace=True)

    return results


def cal_volatility(data, start, end):
    r_std = []
    for i in results.columns:
        residual = []
        R = ts_pro.daily(ts_code=i, start_date=start, end_date=end)
        R.sort_values(by="trade_date", ascending=True, inplace=True)
        R = (R["pct_chg"]/100-0.035).tolist()
        for r in range(243):
            s = R[r] - results_T['Alpha'][i] -\
                data['SMB'][r]*results_T['SMB'][i] -\
                data['HML'][r]*results_T['HML'][i] -\
                data['MKT'][r]*results_T['MKT'][i]
            residual.append(s)
        r_std.append(np.std(residual, ddof=1))
    results_T['std'] = r_std
    return results_T


start = '20210101'
end = '20220101'
date = '20210601'
stocks_codes = get_SZ50_stocks(start, end)
stocks = stocks_codes
df_stocks = group_stocks(stocks, date)
stocks = df_stocks
df_groups = groups_return(stocks, start, end)
data = df_groups
data = SMB_HML(data)
stocks_codes = get_SZ50_stocks(start, end)
data = selection(data, start, end, stocks_codes)
df_final = data
results = OLS(df_final)
results_values = results.values.T
results_T = pd.DataFrame(data=results_values, index=results.columns, columns=['Alpha', 'SMB', 'HML', 'MKT'])
results_T = cal_volatility(data, start, end)
print(results_T)
fig = plt.figure()
plt.scatter(results_T['std'], results_T['Alpha'])
plt.show()