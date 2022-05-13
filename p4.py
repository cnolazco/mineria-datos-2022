import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("spotify.csv",)

df = df.drop(df.tail(3).index)
df["added"] = df["added"].str.replace("‑", "-")
df["added"] = pd.to_datetime(df["added"], format="%Y-%m-%d")
df["year released"] = df["year released"].astype(int)
df["top year"] = df["top year"].astype(int)


mean_duration = df.groupby("artist").mean()["dur"]
mean_duration = mean_duration.sort_values(ascending=False).head(10)
mean_duration.plot(kind="barh", title="Top 10 cantantes con mayor duración promedio por canción", ylabel="seconds", figsize=(15, 5))
plt.savefig("imgs/p4/mean_duration.png")
plt.close()

df_2019 = df[df["year released"] == 2019]["top genre"]
df_2019 = df_2019.value_counts()
df_2019.plot(kind="pie", title="Distribución de géneros en 2019", ylabel="", rotatelabels=True)
plt.savefig("imgs/p4/genres_2019.png")
plt.close()

bottom_rated = df[df["year released"] == 2018]
bottom_rated = df.groupby("artist").sum()["pop"]
bottom_rated = bottom_rated.sort_values().head(10)
bottom_rated.plot(kind="barh", title="Top 10 cantantes menos populares de 2018", ylabel="seconds", figsize=(15, 5))
plt.savefig("imgs/p4/bottom_rated.png")
plt.close()
