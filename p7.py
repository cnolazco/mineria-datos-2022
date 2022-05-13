import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers


def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][df.index[0]], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

def linear_regression(df: pd.DataFrame, x:str, y: str)->dict[str, float]:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(list(df[y]),sm.add_constant(fixed_x), alpha=0.05).fit()
    bands = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]
    #print_tabulate(pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0])
    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    r_2_t = pd.read_html(model.summary().tables[0].as_html(),header=None,index_col=None)[0]
    return {'m': coef.values[1], 'b': coef.values[0], 'r2': r_2_t.values[0][3], 'r2_adj': r_2_t.values[1][3], 'low_band': bands['[0.025'][0], 'hi_band': bands['0.975]'][0]}

def plt_lr(df: pd.DataFrame, x:str, y: str, m: float, b: float, r2: float, r2_adj: float, low_band: float, hi_band: float, colors: tuple[str,str]):
    fixed_x = transform_variable(df, x)
    plt.plot(df[x],[ m * x + b for _, x in fixed_x.items()], color=colors[0])
    plt.fill_between(df[x],
                     [ m * x  + low_band for _, x in fixed_x.items()],
                     [ m * x + hi_band for _, x in fixed_x.items()], alpha=0.2, color=colors[1])


df = pd.read_csv("spotify.csv",)

df = df.drop(df.tail(3).index)
df["added"] = df["added"].str.replace("‑", "-")
df["added"] = pd.to_datetime(df["added"], format="%Y-%m-%d")
df["year released"] = df["year released"].astype(int)
df["top year"] = df["top year"].astype(int)


df_dpop = df.groupby(df["added"].dt.date).mean()
df_dpop = df_dpop.reset_index()
df_dpop = df_dpop.tail(50)[["added", "dnce"]]

df_dpop.plot(kind="scatter", x="added", y="dnce")
a = linear_regression(df_dpop, "added", "dnce")
plt_lr(df=df_dpop, x="added", y="dnce", colors=("red", "orange"), **a)
a = linear_regression(df_dpop.tail(5), "added", "dnce")
plt_lr(df=df_dpop.tail(), x="added", y="dnce", colors=("red", "orange"), **a)
df_dpop_4 = df_dpop[pd.to_datetime(df_dpop["added"]).dt.dayofweek == 4]
a = linear_regression(df_dpop_4, "added", "dnce")
plt_lr(df=df_dpop.tail(), x="added", y="dnce", colors=("blue", "blue"), **a)
plt.xticks(rotation=90)
plt.savefig("imgs/p7/lr_dnce_added.png")
plt.close()

df2 = df_dpop.loc[(pd.to_datetime(df_dpop["added"])>='2020-01-01') & (pd.to_datetime(df_dpop["added"]) < '2022-04-01')]
dfs = [
    ('50D', df_dpop),
    ('10D', df_dpop.tail(10)),
    ('5D', df_dpop.tail()),
    ('jueves', df_dpop[pd.to_datetime(df_dpop["added"]).dt.dayofweek == 4]),
    ('50D-1Y', df2),
    ('10D-Y', df2.tail(10)),
    ('5D-Y', df2.tail(5)),
    ('jueves-Y', df2[pd.to_datetime(df2["added"]).dt.dayofweek == 4]),
]
lrs = [(title, linear_regression(_df, x="added", y="dnce"), len(_df)) for title, _df in dfs]
lrs_p = [(title, lr_dict["m"]*size  + lr_dict["b"], lr_dict) for title, lr_dict, size in lrs]
print(lrs_p)
