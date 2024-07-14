# This script selects a model 

# Load Libraries and Set Options====================================================================
## Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import inflection


## Options 
pd.options.display.max_columns = 20
pd.options.display.max_rows = 20
pd.options.display.max_colwidth = 80
np.set_printoptions(precision=4, suppress=True)


# Source Functions and Objects and Import Data======================================================
## Change wd
from pathlib import Path
import os
Path.cwd() #return wd; PosixPath('/Users/keithpost')
root = '/Users/keithpost/Documents/Python/Python projects/imdb_ml_py/'
os.chdir(root + 'code') #change wd
Path.cwd() #returns new wd


## Functions (and objects)
from _00_helper_fns import get_rec_cosine, get_rec_knn, evaluate_rec


## Data
os.chdir(root + 'data') 
#clean data file
file = open('data_final.pkl', 'rb')
df0 = pickle.load(file) 

#movie title crosswalk
df_title0 = pd.read_csv('movies_metadata.csv')
#convert to snake_case
df_title0.columns = df_title0.columns.map(inflection.underscore) 

#user data
df_user0 = pd.read_csv('ratings_small.csv')
#convert to snake_case
df_user0.columns = df_user0.columns.map(inflection.underscore) 



# Model Selection===================================================================================
## Prep data
df = df0.drop(['title', 'year'], axis=1)
df.info()


## Cosine similarity matrix approach
### Develop cosine similarity matrix
cosine_sim = cosine_similarity(df)


### Run function to get recommendation
get_rec_cosine(df=df0, title='The Matrix', mat=cosine_sim, n=5)
get_rec_cosine(df=df0, title='Forrest Gump', mat=cosine_sim, n=5)
get_rec_cosine(df=df0, title='Apollo 13', mat=cosine_sim, n=5)
get_rec_cosine(df=df0, title='Training Day', mat=cosine_sim, n=5)
get_rec_cosine(df=df0, title='Dances with Wolves', mat=cosine_sim, n=5)
get_rec_cosine(df=df0, title='The Martian', mat=cosine_sim, n=5)

#Drishyam is the only movie in here twice because it has two different years associated with it
get_rec_cosine(df=df0, title='Drishyam', mat=cosine_sim, n=5) #defaults to 2013 version
get_rec_cosine(df=df0, title='Drishyam', year=2013, mat=cosine_sim, n=5)
get_rec_cosine(df=df0, title='Drishyam', year=2015, mat=cosine_sim, n=5)



## KNN approach
### Fit KNN model
features = df.columns

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(df[features])

### Run function to get recommendation (same result as other recommender b/c using same distance matrix)
get_rec_knn(df=df0, title='Star Wars', model=knn, n=5)
get_rec_knn(df=df0, title='Training Day', model=knn, n=5)
get_rec_knn(df=df0, title='Drishyam', model=knn, n=5)
get_rec_knn(df=df0, title='Drishyam', year=2013, model=knn, n=5)
get_rec_knn(df=df0, title='Drishyam', year=2015, model=knn, n=5)


# Model Evaluation==================================================================================
## Check whether there's a movie of same title with different years
df0['title'].duplicated().sort_values(ascending=False) #index 136
df0.iloc[136] #title = "Drishyam"
df0[df0['title']=='Drishyam'] #same title, different years
#Yes...need to retain both title and release year (if we want to be ultra-specific)


## Wrangle clean imdb DF and df_title0 then join them
#isolate cols
df_imdb_ty = df0[['title', 'year']]

#extract year from release_date and place into new column                                                                      
df_title0['year'] = df_title0['release_date'].str[0:4].astype(float)

#merge DFs
df_title_year_id = df_title0.merge(df_imdb_ty, on=['title', 'year'])[['title', 'year', 'id']]
df_title_year_id.info()
#718 x 2 df--'lost' 282 movies

df_join_check = df_imdb_ty.merge(df_title0,
                                 on=['title', 'year'],
                                 how='left')[['title', 'year', 'id']]
                                 
#find what didn't join
unjoined_titles = df_join_check[df_join_check['id'].isna()]['title'].tolist()
len(unjoined_titles)
unjoined_titles[0:5]
#however, many of these are due to poor joining (e.g., punctuation, capaitalization), but there's 
  #more than enough movies available for testing the algorithm
  

## Join title_year_id and user DFs together
#wrangle title_year_id and user data 
df_title_year_id = df_title_year_id.rename(columns={'id': 'movie_id'})

df_user = df_user0.drop('timestamp', axis=1)
df_user['movie_id'] = df_user['movie_id'].astype(str)

#join them together--filtering, inner join
df_title_user = df_user.merge(df_title_year_id, on='movie_id')
df_title_user.info()
#user_id, movie_id, user-specific movie rating, movie title, and movie release year


## Generate DF of similar movies by running algorithm
### Filter df0 for 718 movies that we are evaluating
df_filt = df0.merge(df_title_year_id.drop('movie_id', axis=1), on=['title', 'year'])


### Run function over every movie and store into dictionary
#run once
get_rec_cosine(df=df_filt, title='The Matrix', mat=cosine_sim, n=5) #full output
get_rec_cosine(df=df_filt, title='The Matrix', mat=cosine_sim, n=5, title_only=True) #list of titles

#generate a new cosine similarity matrix
cosine_sim_filt = cosine_similarity(df_filt.drop(['title', 'year'], axis=1))

#set up df for taking on recommended movies
df_recs = df_filt.copy()

df_recs['recs'] = df_filt['title'].apply(lambda x: get_rec_cosine(df=df_filt, title=x, 
                                                                  mat=cosine_sim_filt, n=5, title_only=True))
df_recs = df_recs[['title', 'recs']]
df_recs.head()


## Isolate cases where input movie rated highly and the ratings of other movies watched                                     
### Run once  
#get recs from inputting 'Titanic'
t_alg_titanic_recs = get_rec_cosine(df=df_filt, title='Titanic', mat=cosine_sim_filt, n=10, title_only=True)

#get user_ids of users who watched and rated Titanic highly
df_title_user[df_title_user['title']=="Titanic"] #returns all instances in which Titanic watched
df_titanic_high = df_title_user[(df_title_user['title']=="Titanic") & (df_title_user['rating']>=4)] #Titanic watched and rated highly (>=4)
t_titanic_high_user = df_titanic_high['user_id'].tolist() #isolate users who rated Titanic highly

#return movies watched by users who watched and rated Titanic highly and in recommendation list by algorithm
df_titanic_high_movies = df_title_user[df_title_user['user_id'].isin(t_titanic_high_user)] #all movies watched by users who rated Titanic highly
df_titanic_high_movies_recd = df_titanic_high_movies[df_titanic_high_movies['title'].isin(t_alg_titanic_recs)]

#generate algo-rec'd movies by user (user_id)
df_titanic_high_movies_recd['al4'] = df_titanic_high_movies_recd['rating'] >= 4
df_titanic_rec_prop = df_titanic_high_movies_recd['al4'].agg(['count', 'sum'])
df_titanic_rec_prop['prop'] = df_titanic_rec_prop['sum']/df_titanic_rec_prop['count']
df_titanic_rec_prop.name = 'Titanic'
df_titanic_rec_prop = df_titanic_rec_prop.reset_index()


### Iterate over whole DF
#### Run function for one movie
evaluate_rec(df_feat=df_filt, mat=cosine_sim_filt, n=10, sel_title="Titanic", 
             df_title_user=df_title_user, thresh=4)


### p@5
#### Run function over every movie
df_recs_prop5 = df_filt['title'].apply(lambda x: evaluate_rec(df_feat=df_filt, mat=cosine_sim_filt, n=5, 
                                      sel_title=x, df_title_user=df_title_user, thresh=4))
df_recs_prop5.index = df_filt['title']
df_recs_prop5.info()


#### Compute recommendations
#evaluation by movie
df_recs_prop5['prop'].mean() #0.593

#evaluation by instance
df_recs_prop5['sum'].sum()/df_recs_prop5['count'].sum() #0.633


### p@10
#### Run function over every movie
df_recs_prop10 = df_filt['title'].apply(lambda x: evaluate_rec(df_feat=df_filt, mat=cosine_sim_filt, n=10, 
                                      sel_title=x, df_title_user=df_title_user, thresh=4))
df_recs_prop10.index = df_filt['title']
df_recs_prop10.info()


#### Compute recommendations
#evaluation by movie
df_recs_prop10['prop'].mean() #0.612

#evaluation by instance
df_recs_prop10['sum'].sum()/df_recs_prop10['count'].sum() #0.606


### p@20
#### Run function over every movie
df_recs_prop20 = df_filt['title'].apply(lambda x: evaluate_rec(df_feat=df_filt, mat=cosine_sim_filt, n=20, 
                                      sel_title=x, df_title_user=df_title_user, thresh=4))
df_recs_prop20.index = df_filt['title']
df_recs_prop20.info()


#### Compute recommendations
#evaluation by movie
df_recs_prop20['prop'].mean() #0.613

#evaluation by instance
df_recs_prop20['sum'].sum()/df_recs_prop20['count'].sum() #0.590
