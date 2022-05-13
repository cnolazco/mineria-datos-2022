import pandas as pd


df = pd.read_csv("spotify.csv",)

df = df.drop(df.tail(3).index)
df["added"] = df["added"].str.replace("‑", "-")
df["added"] = pd.to_datetime(df["added"], format="%Y-%m-%d")
df["year released"] = df["year released"].astype(int)
df["top year"] = df["top year"].astype(int)


mean_duration = df.groupby("artist").mean()["dur"]
print("Duración promedio (segundos) de canciones por artista/grupo:\n")
print(mean_duration)

total_duration = df.groupby("artist").sum()["dur"]
print("Duración total (segundos) por por artista/grupo cantando\n")
print(total_duration)

easiest_song = df["dnce"].idxmax()
print("La canción más fácil de bailar:\n")
print(df.iloc[easiest_song])

lowest_dB = df["dB"].idxmin()
print("La canción con menos dB (menos ruidosa):\n")
print(df.iloc[lowest_dB])
