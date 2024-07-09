# Custom Functions for app==========================================================================
## Load modules
import pandas as pd


## Function to generate info on selected movie
def get_movie_info(df, title):
  cols = ['title', 'year', 'Genre', 'Director', 'Starring']
  
  df1 = df[df['title']==title]
  df1 = df1[cols].rename(columns={'title': 'Movie Title',
                                  'year': 'Year Released',
                                  'Genre': 'Genre(s)'})
  
  return df1


## Function to generate recommended movies
def get_rec_cosine(df, title, mat, year=None, n=10, s_score=False, i_rank=False):
  #conditional logic for presence/absence of year
  if year is None:
    idx = df[df['title'] == title].index[0] #finds index of movie in df
  else:
    idx = df[(df['title'] == title) & (df['year'] == year)].index[0] #finds index of movie in df
    
  #generate similarity scores
  sim_scores = list(enumerate(mat[idx])) #create list of similarity scores for given movie w/other movies
  sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #sort scores in desc order
  sim_scores = sim_scores[1:n + 1] #select top 10 (by default) except first one
  movie_indices = [i[0] for i in sim_scores] #extract indices of most similar movies

  #create list objs
  cols1 = ['title', 'year', 'Genre', 'Director', 'Starring']
  cols2 = cols1 + ['similarity_score', 'movie_index']
  
  #generate DF of movie_index and movie_title, year, genre, director, and star1
  df1a = df[cols1].iloc[movie_indices].reset_index().rename(columns={'index': 'movie_index'})
  
  #generate DF of movie_index and similarity_score
  df1b = pd.DataFrame(sim_scores, columns = ['movie_index', 'similarity_score'])

  #join above two DFs together
  df1 = df1a.merge(df1b, on='movie_index').round({'similarity_score': 3})
  df1 = df1
  df1 = df1[cols2]
  # df1 = df1[['movie_title', 'similarity_score', 'movie_index']] #reorder rows
  df1 = df1.rename(columns={'title': 'Movie Title',
                            'year': 'Year Released',
                            'Genre': 'Genre(s)',
                            'similarity_score': 'Similarity Score',
                            'movie_index': 'IMDB Rank'})
  df1.insert(loc=0, column='Rec Rank', value=range(1, len(df1)+1))
  
  #conditional display of results
  if s_score and i_rank:
    df2 = df1.copy()
  elif s_score and not i_rank:
    df2 = df1.drop('IMDB Rank', axis=1)
  elif i_rank and not s_score:
    df2 = df1.drop('Similarity Score', axis=1)
  else:
    df2 = df1.drop(['Similarity Score', 'IMDB Rank'], axis=1)
  

  return df2
