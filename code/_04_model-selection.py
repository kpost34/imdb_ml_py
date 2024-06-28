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
from _00_helper_fns import get_rec_cosine


## Data
os.chdir(root + 'data') 
file = open('data_final.pkl', 'rb')
df0 = pickle.load(file) 



# Model Selection===================================================================================
## Prep data
df = df0.drop('title', axis=1)
df.info()


## Cosine similarity matrix approach
### Develop cosine similarity matrix
cosine_sim = cosine_similarity(df)


### Run function to get recommendation
get_rec_cosine(df=df0, title='Goodfellas', cosine_sim=cosine_sim, n=5)








