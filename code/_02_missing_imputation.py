# This script imputes missing data

# Load Libraries and Set Options====================================================================
## Load libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer


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


## Functions 
from _00_helper_fns import littles_mcar_test


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
cats = ['certificate', 'director', 'star1', 'star2', 'star3', 'star4']

df_mcar = df0.copy()

df_mcar[cats] = df_mcar[cats].astype('category')
df_mcar.drop(['title', 'genre', 'overview'], axis=1, inplace=True)

df_mcar.info() #no objects left

littles_mcar_test(df_mcar) #p = 0, X^2 = 1.78e21; MAR



# Imputation========================================================================================
## Copy DF to new name for section
df_imp = df_mcar.copy()


## Impute categorical variable using most frequent category
#create array
ar_cert = df_imp['certificate'].values.reshape(-1, 1) 

#impute data
cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent') #create imputer
ar_cert_imp = cat_imputer.fit_transform(ar_cert) #impute missing values
cert_imp = pd.Series(ar_cert_imp.flatten(), name='certificate') #convert back to series
cert_imp

#join back into DF
df_imp = pd.concat([df_imp.drop('certificate', axis=1), cert_imp], axis=1)


## Impute numerical variables via KNN
### Convert one-hot encoding
df_imp = pd.get_dummies(df_imp) 


### Create lists of missing cols
cols_miss = ['certificate', 'meta_score', 'gross']


### Impute using KNN 
knn_imputer = KNNImputer(n_neighbors=5)
ar_imp = knn_imputer.fit_transform(df_imp)
#creates an array


### Return data to DF
df_imp2 = pd.DataFrame.from_records(ar_imp)
df_imp2.columns = df_imp.columns #add columns back
df_imp2


## Check for missingness
df0.info() #data import--certificate, meta_score, and gross having missing values
df_imp2_nmiss = df_imp2.isna().sum().reset_index()
df_imp2_nmiss[df_imp2_nmiss['index'].isin(cols_miss)] #0s--all imputed


## Take imputed columns and join back with rest of data
### Reverse one-hot encoding
#isolate non-dummy data
cols_dummy = ['certificate', 'director', 'star1', 'star2', 'star3', 'star4']
cols_remove = cols_dummy.copy()
cols_describe = ['title', 'genre', 'overview']
cols_remove.extend(cols_describe)
cols_ndummy = df0.drop(cols_remove, axis=1).columns.tolist()

#reverse dummy data
cols_dummy_wide = df_imp2.drop(cols_ndummy, axis=1).columns

df_dummy = pd.from_dummies(df_imp2.drop(cols_ndummy, axis=1), sep="_")


### Replace dummy cols with the categorical/string versions
df = pd.concat([df0[cols_describe], #title, genre, overview
                df_imp2.drop(cols_dummy_wide, axis=1), #num cols, genre wide cols, overview wide cols
                df_dummy], axis=1) #dir & star cols in long format
df.info()



# Save Data to File=================================================================================
#save in pickle format to retain data types and categories
afile = open('data_impute.pkl', 'wb')
pickle.dump(df, afile)
afile.close()





