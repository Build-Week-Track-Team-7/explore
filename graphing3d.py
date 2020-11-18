import pandas as pd
import plotly.express as px
from joblib import load
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# importing the csv file to create models.
music_df = pd.read_csv('data.csv', index_col=['name'])
# droping columns that will just muddy up the data
music_df = music_df.drop(columns=['artists', 'release_date', 'id'])
# scalling the information to find the best fit
music_scale = pd.DataFrame(data=StandardScaler().fit_transform(music_df),
                           columns=music_df.columns,
                           index=music_df.index)
# creating cluster defenition to call back when running kmeans
pipeline = load('pipeline.joblib')
# prepare the lables of Kmeans for graphing
labels = pipeline.labels_
# adding lables to color code each group type
music_df['group'] = labels.astype(str)
# sampling the dataset to make a more managable graph
music_sample = music_df.sample(5000, random_state=42)
# color dictionary
colors = {'0': 'crimson',
          '1': '#4da6ff',
          '2': '#ff66cc',
          '3': '#993366',
          '4': '#ffff66',
          '5': '#99ff33',
          '6': '#009933',
          '7': '#00ffff',
          '8': '#cc3300',
          '9': '#cc33ff'}
# graphing the Kmeans test
fig = px.scatter_3d(music_sample,
                    x='energy',  # the first important feature
                    y='danceability',  # the second important feature
                    z='liveness',  # the third important feature
                    color='group',  # calling the K-means group labels
                    color_discrete_map=colors,  # using the color dictionary
                    width=700,
                    height=500,
                    hover_name=music_sample.index)  # show the name of the song
fig.update_traces(marker=dict(size=1),  # changing the dot size
                  showlegend=False)  # removing the ledgend
fig.show()
