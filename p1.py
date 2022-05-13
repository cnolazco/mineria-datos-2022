from kaggle.api import kaggle_api_extended as kaggle
import os


api = kaggle.KaggleApi()
api.authenticate()

api.dataset_download_file(
  "muhmores/spotify-top-100-songs-of-20152019",
  "Spotify 2010 - 2019 Top 100.csv"
)

os.rename("./Spotify%202010%20-%202019%20Top%20100.csv", "./spotify.csv")
