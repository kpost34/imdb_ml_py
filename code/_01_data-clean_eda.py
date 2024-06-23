

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



# Source Functions and Import Data==================================================================
## Change wd
from pathlib import Path
import os
Path.cwd() #return wd; PosixPath('/Users/keithpost')
root = '/Users/keithpost/Documents/Python/Python projects/imdb_ml_py/'
os.chdir(root + 'code') #change wd
Path.cwd() #returns new wd


## Functions
#insert function script here


## Data
os.chdir(root + 'data') 
df0 = pd.read_csv("imdb_top_1000.csv") 
#convert to snake_case
df0.columns = df0.columns.map(inflection.underscore) 



# Initial Cleaning==================================================================================
## Get basic info
df0.shape #1000 x 16
df0.columns
df0.index
df0.dtypes 

df0.head()
df0.describe()
df0.info()


## Cleaning
### Remove poster_link, rename cols
df0 = df0.drop(columns=['poster_link']).rename(columns={"series_title": "title",
                                                        "runtime": "runtime_min"})
                                                        
### Remove " min" in runtime col and convert to int
df0['runtime_min'] = df0['runtime_min'].replace(to_replace=' min$', value='', regex=True).astype(int)


### Convert gross to float
df0['gross'] = df0['gross'].replace(',', '', regex=True).astype(float)


### Convert released_year to int...but need to replace mistake in data
df0['released_year'].unique() #'PG' -- mistake, so let's replace
df0[df0['released_year']=='PG'] #Apollo 13, released in 1995
df0['released_year'] = df0['released_year'].replace('PG', '1995', regex=True)

df0[df0['released_year']=='PG'] #empty, so updated made

df0['released_year'] = df0['released_year'].astype(int)


### Convert genre into dummy variables & join back to DF
#conversion
genre_dummies = df0.genre.str.get_dummies(", ").rename(columns=str.lower)
genre_dummies.columns

#join back
df0 = df0.join(genre_dummies.add_prefix("genre_"))
df0.columns



# EDA===============================================================================================
df = df0.copy() 

colors = ['blue', 'green', 'red', 'purple', 'orange', "grey"]
nums = ['released_year', 'runtime_min', 'imdb_rating', 'meta_score', 'no_of_votes', 'gross']
cats_n = ['director', 'star1', 'star2', 'star3']

## Univariate
### Histograms 
#### Numerical fields
fig, axes = plt.subplots(2, 3)
n = 0
for i in range(0, 2):
  for j in range(0, 3):
    axes[i, j].hist(df[nums[n]], color=colors[n], edgecolor="Black")
    axes[i, j].set(xlabel=nums[n], ylabel="count")
    n = n + 1

plt.tight_layout()    
plt.show() 
plt.close() 


### Counts of director and stars
df_cats_n = df[cats_n]
director_stars_vcs = {col: df_cats_n[col].value_counts() for col in df_cats_n.columns}

#plot histogram of the counts
plt.figure(figsize=(12, 10))

for i, (col, counts) in enumerate(director_stars_vcs.items(), 1):
    plt.subplot(2, 2, i)  # Create a 2x2 subplot grid and select the i-th subplot
    plt.hist(counts.values, bins=range(1, counts.max() + 2), color=colors[i], edgecolor='black', 
             align='left')
    plt.title(f'Histogram of Value Counts for {col}')
    plt.xlabel('Counts')
    plt.ylabel('Frequency')
    plt.xticks(range(1, counts.max() + 1))

plt.subplots_adjust(wspace=.4, hspace=0.2)

plt.show()
plt.close()


### Barplots
#certificates
df_cert_n = df['certificate'].value_counts().reset_index()

plt.bar(df_cert_n['certificate'], df_cert_n['count'], edgecolor='black')
plt.xticks(rotation=60)
plt.show()
plt.close()


### Genre dummy variables
#creates DFs and labels
gen_dum1 = genre_dummies.iloc[:, 0:12]
gen_dum2 = genre_dummies.iloc[:, 12:]

gen_dum1_nm = gen_dum1.columns.to_numpy()
gen_dum2_nm = gen_dum2.columns.to_numpy()

gen_colors = ['blue', 'green'] * 6

#generate first set of plots
fig, axes = plt.subplots(3, 4)

n = 0
for i in range(0, 3):
  for j in range(0, 4):
    var = gen_dum1_nm[n]
    axes[i, j].bar(x=gen_dum1[var].value_counts().index,
                   height=gen_dum1[var].value_counts(),
                   color=gen_colors[n], edgecolor='black')
    axes[i, j].set_ylim(0, 1000)
    axes[i, j].set_xlabel(var)
    fig.supylabel("Count")
    n = n + 1

plt.subplots_adjust(wspace=.5, hspace=0.2)

plt.show()
plt.close()

#generate second set of plots
fig, axes = plt.subplots(3, 3)

n = 0
for i in range(0, 3):
  for j in range(0, 3):
    var = gen_dum2_nm[n]
    axes[i, j].bar(x=gen_dum2[var].value_counts().index,
                   height=gen_dum2[var].value_counts(),
                   color=gen_colors[n], edgecolor='black')
    axes[i, j].set_ylim(0, 1000)
    axes[i, j].set_xlabel(var)
    fig.supylabel("Count")
    n = n + 1

plt.subplots_adjust(wspace=.3, hspace=0.2)

plt.show()
plt.close()


## TF-IDFs from overviews
df['overview'].head(3)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_mod = TfidfVectorizer() #create tfidf model
X = tfidf_mod.fit_transform(df['overview']) #fit model to data
df_tfidf = pd.DataFrame(X.toarray(), columns=tfidf_mod.get_feature_names_out()) #populate df of TF-IDFs

#next steps
#1) calculate average TF-IDF score across all documents (which can be used as the term's
  #imortance and distinctiveness across the corpus)
#2) sort these terms (columns) based on avg TF-IDF scores in descending order
#3) choose the top N terms from the sorted list--N selected using specific requirements and
  #constraints--either a fixed number or percentage
#4) these top N terms are considered the most important and distinctive features based on TF-IDF
  #scores and thus are the most relevant for distinguishing between documents in your corpus
#5) construct a new feature matrix using only the top N terms selected in the previous step


## Bivariate
















## Explore one-hot encoding
#star cols
df0.info()
star1_counts = df0.star1.value_counts().sort_values(ascending=False) #8-12 + Other
star2_counts = df0.star2.value_counts().sort_values(ascending=False)[0:20] #4-7 + Other
star3_counts = df0.star3.value_counts().sort_values(ascending=False)[0:20] #4-5 + Other

# Creating a histogram
plt.hist(star1_counts, bins=12, edgecolor='black')

# Adding titles and labels
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Showing the plot
plt.show()
plt.close()


#certificate
#U = unrestricted for public exhibition and family-friendly
#A = adults only
#UA = unrestricted but parental discretion for children < 12
#R = requires adult if < 17
#PG - 13: some material inappropriate for children under 13
#PG = some material may not be suitable for children
#Passed
#G = all ages admitted
# Approved = pre-1968 titles - deemed 'moral'
#TV-PG
#GP
#TV-14
#16
#TV-MA - adult
#Unrated
#U/A



# Explore Data======================================================================================





## Descriptive stats
df_num = df0[['imdb_rating', 'meta_score', 'no_of_votes']]

df_num.sum()
df_num.mean()

df_num.corr()
df_num.cov()


## Unique values
df0['genre'].unique()
df0['genre'].value_counts()


## Summarize data



## Visualizations



## Assess missingness
#by row
df0.dropna() #286 rows with at least 1 NA (start with 1000, now 714)
df0.dropna(how="all") #0 rows with ALL NAs
df0.dropna(thresh=16) #thresh = # of non-NAs required (16 cols so all need to be NA); here
  #714 remain (same as df0.dropna())--286 rows w/1+ NAs
df0.dropna(thresh=15) #894 = 106 rows w/2+ NAs
df0.dropna(thresh=14) #965 = 35 rows w/3+ NAs
df0.dropna(thresh=13) #1000 (no row w/4+ NAs)
#714 with 0 NAs, 180 with 1 NA, 71 with 2 NAs, 35 with 3 NAs

#by column
df0.series_title.isna() #returns Boolean Series of whether value is NA or not
df0.dropna(axis="columns") #drops 3 cols that have at least 1 NA (imdb_rating, meta_score, gross)
df0.dropna(axis="columns", how="all") #no change b/c no col has all NAs


## Data transformation
#duplicates
df0.duplicated() #returns T/F on whether it's a duplicate
df0.drop_duplicates() #1000, so none

#rename() to create transformed version of dataset
df0.rename(columns=str.upper) #cols are in uppercase
df0.rename(columns={"series_title": "Series"}) #series_title renamed to Series


## Discretization and binning
df0.info()
df0.head()







