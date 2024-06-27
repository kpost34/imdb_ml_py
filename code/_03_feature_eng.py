# This script engineers features for machine learning

# Load Libraries and Set Options====================================================================
## Load libraries
import pandas as pd
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
from _00_helper_fns import group_categories


## Data
os.chdir(root + 'data') 
file = open('data_impute.pkl', 'rb')
df0 = pickle.load(file) 



# Feature Selection=================================================================================
## Extraneous variables
objs = ['genre', 'overview']

df = df0.drop(objs, axis=1)


## Multicollinearity
#isolate list of numeric columns
dir_stars = ['director', 'star1', 'star2', 'star3', 'star4']
nnums = objs + dir_stars + ['certificate']
nums = df0.drop(nnums + ['title'], axis=1).columns.tolist()

#perform correlations
df_corr = df[nums].corr(method='spearman')
df_corr[df_corr > 0.9]
#no pairs are highly correlated



# Feature Engineering: Rare-label Encoding==========================================================
## certificate
#assess frequencies
cats_cert = df['certificate'].unique().tolist()
df['certificate'].value_counts()
#2% threshold, so < 20 = 'Other'

#develop mapping
rare_certs = df['certificate'].value_counts()
rare_certs = rare_certs[rare_certs >20].index.tolist()

#implement encoding
df['certificate'] = df['certificate'].apply(group_categories, rare_cats=rare_certs, new_cat='Other')


## director and stars
## Explore one-hot encoding (and rare-label encoding)
### Generate DFs for star & dir cols
star1_counts = df.star1.value_counts().sort_values(ascending=False).to_frame() #8-12 + Other
star2_counts = df.star2.value_counts().sort_values(ascending=False).to_frame() #4-7 + Other
star3_counts = df.star3.value_counts().sort_values(ascending=False).to_frame() #4-5 + Other
star4_counts = df.star4.value_counts().sort_values(ascending=False).to_frame() #2-3 + Other
dir_counts = df.director.value_counts().sort_values(ascending=False).to_frame() #8-14 + Other

#append new column 'col' and populate with index name
star1_counts['col'] = star1_counts.index.name
star2_counts['col'] = star2_counts.index.name
star3_counts['col'] = star3_counts.index.name
star4_counts['col'] = star4_counts.index.name
dir_counts['col'] = dir_counts.index.name

#combine dfs by row
df_stars_dir_n = pd.concat([star1_counts, star2_counts, star3_counts, star4_counts, dir_counts])

### Create histograms
colors = ['blue', 'green', 'red', 'purple', 'orange', 'black']
stars_dir = ['star1', 'star2', 'star3', 'star4', 'director']

fig, axes = plt.subplots(2, 3)

for i, variable in enumerate(stars_dir):
  #create each df by filtering
  df_plot = df_stars_dir_n[df_stars_dir_n['col']==variable]
  
  #create row and col indexes
  row = i // 3
  col = i % 3
  
  #select a color
  color=colors[i]
  
  #build the histogram
  axes[row, col].hist(df_plot['count'], color=color, edgecolor='black')
  
  #add labels
  title = 'Histogram of \n' + variable + ' Values'
  axes[row, col].set_title(title)
  axes[row, col].set_xlabel('Count')
  axes[row, col].set_ylabel('Frequency')

#remove 6th subplot
fig.delaxes(axes[1, 2])

#adjust spacing between subplots
# plt.subplots_adjust(wspace=0.05, hspace=0.05)  
plt.tight_layout()
plt.show()
plt.close() 


## Implement encoding
### Rare-label encoding
cats_star1 = star1_counts[0:9].index.tolist() #8-12 + Other
cats_star2 = star2_counts[0:10].index.tolist() #4-7 + Other
cats_star3 = star3_counts[0:8].index.tolist() #4-5 + Other
cats_star4 = star4_counts[0:3].index.tolist() #2-3 + Other
cats_dir = dir_counts[0:2].index.tolist() #8-14 + Other


## Explore rare-label encoding
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
