# Custom Functions for app==========================================================================
## Load modules
import pandas as pd


## Function to generate recommended movies
def get_rec_cosine(df, title, mat, year=None, n=10):
  if year is None:
    idx = df[df['title'] == title].index[0] #finds index of movie in df
  else:
    idx = df[(df['title'] == title) & (df['year'] == year)].index[0] #finds index of movie in df
  sim_scores = list(enumerate(mat[idx])) #create list of similarity scores for given movie w/other movies
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #sort scores in desc order
  sim_scores = sim_scores[1:n + 1] #select top 10 (by default) except first one
  movie_indices = [i[0] for i in sim_scores] #extract indices of most similar movies

  #generate DF of movie_index and movie_title
  df1a = df['title'].iloc[movie_indices].reset_index().rename(columns={'index': 'movie_index',
                                                                        'title': 'movie_title'})
  #generate DF of movie_index and similarity_score
  df1b = pd.DataFrame(sim_scores, columns = ['movie_index', 'similarity_score'])

  #join above two DFs together
  df1 = df1a.merge(df1b, on='movie_index').round({'similarity_score': 3})
  df1 = df1[['movie_title', 'similarity_score', 'movie_index']] #reorder rows
  df1 = df1.rename(columns={'movie_title': 'Movie Title',
                            'similarity_score': 'Similarity Score',
                            'movie_index': 'IMDB Rank'})
  df1.insert(loc=0, column='Rank', value=range(1, len(df1)+1))
  
  return df1
