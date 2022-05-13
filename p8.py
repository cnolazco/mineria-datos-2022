from scipy.stats import mode
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = get_cmap(len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    plt.savefig(file_path)
    plt.close()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neightbors(
    points: list[np.array], labels: np.array, input_data: list[np.array], k: int
):
    input_distances = [
        [euclidean_distance(input_point, point) for point in points]
        for input_point in input_data
    ]
    points_k_nearest = [
        np.argsort(input_point_dist)[:k] for input_point_dist in input_distances
    ]
    return [
        mode([labels[index] for index in point_nearest])
        for point_nearest in points_k_nearest
    ]


df = pd.read_csv("spotify.csv",)

df = df.drop(df.tail(3).index)
df["added"] = df["added"].str.replace("‑", "-")
df["added"] = pd.to_datetime(df["added"], format="%Y-%m-%d")
df["year released"] = df["year released"].astype(int)
df["top year"] = df["top year"].astype(int)


df = df.rename(columns={"top genre": "genre"})
df_class = df[(df["genre"] == "indie pop") | (df["genre"] == "k-pop") | (df["genre"] == "latin")]
df_class = df_class[["dnce", "bpm", "genre"]]
scatter_group_by("imgs/p8/classification.png", df_class, "dnce", "bpm", "genre")
list_t = [
    (np.array(tuples[0:1]), tuples[2])
    for tuples in df_class.itertuples(index=False, name=None)
]
points = [point for point, _ in list_t]
labels = [label for _, label in list_t]

kn = k_nearest_neightbors(
    points,
    labels,
    [np.array([75, 180]), np.array([0, 0]), np.array([85, 50]), np.array([90, 150])],
    5,
)
print(kn)
