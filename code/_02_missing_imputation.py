# This script imputes missing data

# Load Libraries and Set Options====================================================================
## Load libraries
import pandas as pd
import inflection
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


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
#


## Data
os.chdir(root + 'data') 
file = open('data_initial_clean.pkl', 'rb')
df0 = pickle.load(file) 



# Assess Missingness================================================================================
### Exploration
df0.info() #certificate, meta_score, gross have missing data

#by row
df0.dropna() #286 rows with at least 1 NA (start with 1000, now 714)
df0.dropna(how="all") #0 rows with ALL NAs
df0.dropna(thresh=47) #thresh = # of non-NAs required (16 cols so all need to be NA); here
  #714 remain (same as df0.dropna())--286 rows w/1+ NAs
df0.dropna(thresh=46) #894 = 106 rows w/2+ NAs
df0.dropna(thresh=45) #965 = 35 rows w/3+ NAs
df0.dropna(thresh=44) #1000 (no row w/4+ NAs)
#714 with 0 NAs, 180 with 1 NA, 71 with 2 NAs, 35 with 3 NAs

#by column
df0.title.isna() #returns Boolean Series of whether value is NA or not
df0.dropna(axis="columns") #drops 3 cols that have at least 1 NA (imdb_rating, meta_score, gross)
df0.dropna(axis="columns", how="all") #no change b/c no col has all NAs


### Test for MCAR
#make strings (objects) into other forms or drop before running mcar test
df0.info()












