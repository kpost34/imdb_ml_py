#This script contains functions to help with coding

# Load Packages=====================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
import statsmodels.api as sm


# EDA Function======================================================================================
## Function to generate faceted barplots of genres
def make_genres_plot(df, var1, var2):
  df1 = df[[var1, var2]]
  df1_n = df1.groupby([var1, var2], as_index=False).size()
  
  p_var = sns.catplot(x=var1, y="size", kind="bar", hue=var2, data=df1_n)
  p_var.set_axis_labels(x_var=var1, y_var="Number")
  p_var=sns.move_legend(p_var, loc="lower center", ncol=2)
  plt.show()
  plt.close()



# Missingness Function==============================================================================
## Function for Little's MCAR test
def littles_mcar_test(data):
  # Ensure the input is a DataFrame
  if not isinstance(data, pd.DataFrame):
      raise ValueError("Input data must be a pandas DataFrame")
  
  # Remove columns with no missing values
  data = data.loc[:, data.isnull().any()]

  if data.isnull().sum().sum() == 0:
      print("No missing values found. Unable to perform Little's MCAR test.")
      return

  # One-hot encode categorical variables
  data = pd.get_dummies(data, dummy_na=True, drop_first=True)

  n, m = data.shape

  # Generate the missingness indicator matrix
  R = data.isnull().astype(int)

  # Calculate mean and covariance of observed data
  means = data.mean()
  cov = data.cov()

  # Calculate the test statistic
  test_stat = 0
  for i in range(m):
      for j in range(i, m):
          oij = np.sum((1 - R.iloc[:, i]) * (1 - R.iloc[:, j]))
          mij = (1 - means[i]) * (1 - means[j]) * n
          sij = cov.iloc[i, j]
          if sij != 0:
              test_stat += ((oij - mij)**2) / sij

  # Degrees of freedom
  df = (m * (m + 1)) / 2

  # P-value
  p_value = chi2.sf(test_stat, df)

  # Print the result
  print("Chi-square statistic:", test_stat)
  print("P-value:", p_value)

  # Interpret the result
  if p_value < 0.05:
      return "Reject the null hypothesis: Missingness is not completely at random."
  else:
      return "Fail to reject the null hypothesis: Missingness is completely at random."



# Feature Engineering Functions=====================================================================
## Function to group rare categories
def group_categories(variable, rare_cats, new_cat):
  if variable in rare_cats:
    return new_cat
  else:
    return variable


## Function to generate faceted qqplots
def make_qqplots(df, vars, remove_last=False):
  #set fig and subplots
  fig, axes = plt.subplots(2, 3, figsize=(8, 6))

  #plot qqplots for each numerical predictor
  for i, column_name in enumerate(vars):
    row_index = i // 3
    col_index = i % 3
    ax = axes[row_index, col_index] 
    sm.qqplot(df[column_name], line='s', ax=ax)
    ax.set_title(f'{column_name}')
    
  if remove_last:
    axes[1, 2].remove()

  #adjust layout and display plot
  plt.tight_layout()
  plt.show()
  plt.close()



# Modelling Function================================================================================
## Function to recommend movie using cosine similarity matrix
def get_rec_cosine(df, title, mat, year=None, n=10, title_only=False):
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
  
  #title-only output
  if title_only:
    df2 = df1['movie_title'].tolist()
  else: 
    df2 = df1
  
  return df2


## Function to recommend movie using KNN
def get_rec_knn(df, title, model, year=None, n=5):
    #create two DFs: one with and one without movie titles and year
    df_title = df.copy()
    df = df_title.drop(['title', 'year'], axis=1)
    
    #find the index of the movie in df_title
    if year is None:
      idx = df_title[df_title['title'] == title].index[0]
    else:
      idx = df_title[(df_title['title'] == title) & (df_title['year'] == year)].index[0]
    
    #get the feature vector of the movie from df
    movie_features = df.iloc[idx].values.reshape(1, -1)
    
    #ensure movie_features is a DataFrame with column names
    movie_features = pd.DataFrame(movie_features, columns=df.columns)
    
    #find the nearest neighbors
    distances, indices = model.kneighbors(movie_features, n_neighbors = n + 1)
    
    #grab indices of recommended movies
    recd_movie_indices = indices.flatten()[1:]
    
    #get the titles of the recommended movies (excluding the input movie itself)
    df1a = df_title.iloc[recd_movie_indices]['title'].reset_index().rename(columns={"index": "movie_index",
                                                                                    "title": "movie_title"})
    #calculate similarities
    df1b = pd.DataFrame({'similarity_score': 1-distances.flatten()[1:],
                         'movie_index': recd_movie_indices})
    
    #join above two DFs together
    df1 = df1a.merge(df1b, on='movie_index').round({'similarity_score': 3})
    df1 = df1[['movie_title', 'similarity_score', 'movie_index']] #reorder rows
    
    return df1



# Model Diagnostics Function========================================================================
## Returns numbers of highly rated and total movies, their proportion, from users who rated inputted 
  #movie highly
def evaluate_rec(df_feat, mat, n, sel_title, df_title_user, thresh):
  
  #get recs using algo
  t_alg_recs = get_rec_cosine(df=df_feat, title=sel_title, mat=mat, n=n, title_only=True)
  
  #get user_ids of users who watched and rated selected movie highly
  df_title_user[df_title_user['title']==sel_title] #returns all instances in which selected movie watched
  df_title_high = df_title_user[(df_title_user['title']==sel_title) & (df_title_user['rating']>=thresh)] #sel_title watched and rated highly
  t_title_high_user = df_title_high['user_id'].tolist() #isolate users who rated sel_title highly
  
  #return movies watched by users who watched and rated selected movie highly and in recommendation list by algorithm
  df_title_high_movies = df_title_user[df_title_user['user_id'].isin(t_title_high_user)] #all movies watched by users who rated sel_title highly
  df_title_high_movies_recd = df_title_high_movies[df_title_high_movies['title'].isin(t_alg_recs)]
  
  #generate algo-rec'd movies by user (user_id)
  df_title_high_movies_recd['al4'] = df_title_high_movies_recd['rating'] >= thresh
  df_title_rec_prop = df_title_high_movies_recd['al4'].agg(['count', 'sum'])
  df_title_rec_prop['prop'] = df_title_rec_prop['sum']/df_title_rec_prop['count']
  df_title_rec_prop.name = None
  
  return df_title_rec_prop





