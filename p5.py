import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols


df = pd.read_csv("spotify.csv",)

df = df.drop(df.tail(3).index)
df["added"] = df["added"].str.replace("‑", "-")
df["added"] = pd.to_datetime(df["added"], format="%Y-%m-%d")
df["year released"] = df["year released"].astype(int)
df["top year"] = df["top year"].astype(int)


df_years = df.groupby(["top genre", df["year released"]])[["nrgy"]]
df_years = df_years.mean().head(15)
df_years = df_years.reset_index()
df_years = df_years.drop("year released", axis=1)
df_years = df_years.rename(columns={"top genre": "genre"})

df_years.boxplot("nrgy", by="genre", figsize=(5, 10))
plt.xticks(rotation=90)
plt.savefig("imgs/p5/boxplot_nrgy.png")
plt.close()

model = ols("nrgy ~ genre", data=df_years).fit()
df_anova = sm.stats.anova_lm(model, typ=2)

if df_anova["PR(>F)"][0] < 0.005:
    print("Hay diferencias")
    print(df_anova)
else:
    print("No hay diferencias")

print(df_years)
