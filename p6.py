import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers


def transform_variable(df: pd.DataFrame, x: str) -> pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x]  # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])

def linear_regression(df: pd.DataFrame, x, y) -> None:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='green')
    plt.plot(df[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='red')
    plt.xticks(rotation=90)
    plt.savefig(f'imgs/p6/lr_{y}_{x}.png')
    plt.close()


df = pd.read_csv("spotify.csv",)

df = df.drop(df.tail(3).index)
df["added"] = df["added"].str.replace("‑", "-")
df["added"] = pd.to_datetime(df["added"], format="%Y-%m-%d")
df["year released"] = df["year released"].astype(int)
df["top year"] = df["top year"].astype(int)


df_artists = df[(df["top genre"] == "dance pop") & (df["year released"] == 2016)]
df_artists = df_artists.groupby("artist").mean()
df_artists = df_artists.reset_index()
df_artists = df_artists.drop("artist", axis=1)

linear_regression(df_artists, "dnce", "bpm")
linear_regression(df_artists, "nrgy", "val")
linear_regression(df_artists, "live", "dnce")
