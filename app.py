# Load Modules, Functions, and Data=================================================================
## Shiny-related and other functions
from shiny import App, render, ui
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import pickle
import pandas as pd


## Algorithm function
from app_fns import get_rec_cosine


## Data
root = '/Users/keithpost/Documents/Python/Python projects/imdb_ml_py/'
os.chdir(root + 'data')
file = open('data_final.pkl', 'rb')
df0 = pickle.load(file)



# Generate Cosine Similarity Matrix for app=========================================================
#cosine similarity matrix
df = df0.drop(['title', 'year'], axis=1)
cosine_sim = cosine_similarity(df)



# UI================================================================================================
app_ui = ui.page_fluid(
  ui.navset_pill(
    ui.nav_panel("App", 
      ui.tags.br(),
      ui.h2("App content"),
      #inputs
      ui.input_text("txt_movie", "Select a movie"),
      ui.input_slider("sld_n_recs", "Number of recommendations", min=1, max=10, value=5),
      ui.input_checkbox("chk_sim_score", "Similarity score", False),
      ui.input_checkbox("chk_rank", "IMDB rank", False),
      
      #output
      ui.output_data_frame("tab_recs")
    ),
    
    ui.nav_panel("Instructions", 
      ui.tags.br(),
      ui.h2("Instructions info"),
      "--add instructions here--"
    ),
  
    ui.nav_panel("More info", 
      ui.tags.br(),
      ui.h2("Data Preparation"),
      
      ui.p(
        "The ", 
        ui.a("top 1000 movies by IMDB rating", 
        href="https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows"),
        """ were used as the data source. This data file contains the following information: movie title,
        year of release, certificate (rating), runtime, genre, IMDB rating, overview, meta-score,
        director, stars 1-4, and money grossed."""
      ),
      
      ui.p(
        """The movie genre was converted into dummy variables as one ore more genres can apply to a single
        movie. A TF-IDF was performed on the movie overview, removing all stop words, and retaining
        the top 11 words (above 0.01). These were performed prior to imputation as genre was a 
        'composite' variable and overview was unique text."""
      ),
      
      ui.p(
        """The data were assessed for missingness, which was found for certificate, meta-score, and
        grossed money earned. Little's MCAR test was performed, which was significant suggesting
        that missingness was not completely at random. Certificate was imputed using the most
        frequent category. Meta-score and grossed money earned were impute using K-nearest neighbors."""
      ),
      
      ui.p(
        """Multicollinearity was assessed for all numerical fields and no pair was found to be highly
        correlated (spearman rank correlation > 0.9). Rare label encoding was conducted on certificate,
        director, and star fields (4) by retaining the most common categories and binning the remainder
        as "Other". These six features underwent one-hot encoding to generate sets of dummy variables.
        All remaining numerical fields (excluding 0-1 dummy variables, strings, etc.) underwent
        min-max scaling because of non-normal distributions."""
      ),
      
      ui.tags.br(),
      
      ui.h2("Algorithm"),
      
      ui.p("""A cosine similarity matrix was generated using the cleaned, feature-engineered top 1000
      IMDB movies dataset. A custom function called get_rec_cosine() was developed which takes in
      the movie title (and year, if necessary) to generate a dataframe of recommended movies:
        titles, similarity scores, and position in the IMDB top 1000."""),
      
      ui.tags.br(),
      ui.h2("Diagnostics"),
      
      ui.p("--Enter diagnostics information here once it's completed--")
    
    ),
    
    ui.nav_panel("Developer", 
      ui.tags.br(),
      ui.h2("Keith Post"),
      ui.p(
        ui.a("Repo", href="https://github.com/kpost34/imdb_ml_py"),
        
        ui.tags.br(),
        
        ui.a("GitHub", href="https://github.com/kpost34"),
        
        ui.tags.br(),
        
        ui.a("LinkedIn", href="https://www.linkedin.com/in/keith-post/")
      )
    ),
    id="tab"
  )
)



# Server============================================================================================
def server(input, output, session):
  @output
  @render.data_frame
  
  def tab_recs():
    movie_title = input.txt_movie()
    n_recs = input.sld_n_recs()
    
    #check if movie title is provided
    if not movie_title:
      return pd.DataFrame()
    
    #check if movie title is in database
    elif movie_title not in df0['title'].values:
      return pd.DataFrame({'Message': [f"""The movie '{movie_title}' is not in the database. Please 
                            enter a valid movie title."""]})
    
    else:
      df_out = get_rec_cosine(df=df0, 
                              title=movie_title, 
                              mat=cosine_sim, 
                              n=n_recs)
      return df_out


app = App(app_ui, server)


#Updates:
#1) rename the columns to say: "Movie Title", "Similarity Score", and "IMDB Rank"
#2) add a row number field so that it ranks the recommendations from 1 to x
#3) by default, it should return the title as well as the release year and genre (and maybe director
  #and star1?)
#4) add functionality for checkboxes
#5) populate instructions tab
#6) populate more info: diagnostics (once completed)
#7) turn strings into objects that get imported
#8) deploy app
# 
# import pandas as pd
# 
# x = range(5)
# x = range(1, 6)
# for n in x:
#   print(n)
# 
# 
#   
#   
#   
# data = {'A': [10, 20, 30],
#         'B': ["dog", "cat", "mouse"]}
# data = pd.DataFrame(data)
# 
# 
# data.insert(loc=0, column='Rank', value=range(1, len(data)+1))
