# Load Modules, Functions, and Data=================================================================
## Shiny-related and other functions
from shiny import App, render, ui
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import pickle
import pandas as pd


## Algorithm function
from app_fns import get_movie_info, get_rec_cosine


## Data
#clean data
root = '/Users/keithpost/Documents/Python/Python projects/imdb_ml_py/'
os.chdir(root + 'data')
file = open('data_final.pkl', 'rb')
df0 = pickle.load(file)

#original data
df_orig0 = pd.read_csv('imdb_top_1000.csv')



# Data Wrangling====================================================================================
## Clean original data file
cols_retain = ['Series_Title', 'Released_Year', 'Genre', 'Director', 'Star1']

df_orig = df_orig0[cols_retain].rename(columns={'Series_Title': 'title',
                                                'Released_Year': 'year',
                                                'Star1': 'Starring'})

df_orig['year'] = df_orig['year'].replace('PG', '1995', regex=True)

df_orig['year'] = df_orig['year'].astype(float)


## Generate cosine similarity matrix for app
#cosine similarity matrix
df = df0.drop(['title', 'year'], axis=1)
cosine_sim = cosine_similarity(df)


## Join clean df and raw df
df0 = df0.merge(df_orig, on=['title', 'year'])

# 

# UI================================================================================================
app_ui = ui.page_fluid(
  ui.navset_pill(
    ui.nav_panel("App", 
      ui.tags.br(),
      ui.h2("Movie Recommender"),
      
      #inputs
      ui.input_text("txt_movie", "Enter a movie title"),
      ui.input_slider("sld_n_recs", "Number of recommendations", min=1, max=8, value=5),
      ui.input_checkbox("chk_sim_score", "Similarity score?", False),
      ui.input_checkbox("chk_imdb_rank", "IMDB rank?", False),
      
      #output
      ui.h2(ui.output_text("txt_sel")),
      ui.output_data_frame("tab_sel"),
      
      ui.tags.br(),
      
      ui.h2(ui.output_text("txt_recs")),
      ui.output_data_frame("tab_recs")
    ),
    
    ui.nav_panel("Instructions", 
      ui.tags.br(),
      """Please enter the title of a movie in the dialog box. Use the slider to set the number of
      recommended movies returned. The app will always return the following information about the
      movie enered into the box: title, year released, genre(s), director, and first star. By default, 
      the app will return the same information for the recommended movies as well as the recommendation
      rank or 'rec rank' (i.e., strongest to weakest recommendation out of the total list). The 
      similarity score (i.e., from the cosine similarity matrix using engineered features) and
      IMDB Rank (within the top 1000) are optionally returned using the checkboxes."""
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
        frequent category. Meta-score and grossed money earned were imputed using K-nearest neighbors."""
      ),
      
      ui.p(
        """Multicollinearity was assessed for all numerical fields and no pair was found to be highly
        correlated (spearman rank correlation < 0.9 for all comparisons). Rare label encoding was conducted on certificate,
        director, and all four star fields by retaining the most common categories and binning the remainder
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
      
      ui.p("""Precision at k (P@k) was used to evaluate the recommendation algorithm. This metric
      determines how many of the top-k recommendations have high ratings from the same users who
      rated the input movie highly. Out of the 1000 movies in the top IMDB movie list, 718 were used
      in calculating P@k with user ratings data from """,
      ui.a("Kaggle's The Movies Dataset.",
      href="https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset")
      ),
      ui.p(""""Because users can rate the input movie highly and have not watched (and thus rated) the
      top-k recommended movies, it's possible to calculate P@k on a per-movie basis or a per-movie-instance basis. The former means that the proportion or percentage of top-k recommended movies
      rated highly by the group of users who rated the inputted movie highly would be the score 
      that becomes one of the potentially 718 numbers averaged to determine the P@k. The latter
      holds onto the actual fraction (number of top-k recommended movies rated highly over the
      total number of top-k recommended movies watched by the group of users who rated the inputted
      movie highly) when computing the P@k."""),
      ui.p("""The P@k was computed for 5, 10, and 20 recommended movies and using both on a per-movie
      and a per-movie-instance basis:"""),
      ui.HTML("""
        <ul>
          <li>P@5 (per-movie): 0.593</li>
          <li>P@5 (per-movie-instance): 0.633</li>
          <li>P@10 (per-movie): 0.612</li>
          <li>P@10 (per-movie-instance): 0.606</li>
          <li>P@20 (per-movie): 0.613</li>
          <li>P@20 (per-movie-instance): 0.590</li>
        </ul>
        """),
        ui.p("""P@k greater than 0.5 is considered a strong performance, which is accomplished
        when k = 5, 10, or 20 and on both per-movie and per-movie-instance bases.""")
    ),
    
    ui.nav_panel("Developer", 
      ui.tags.br(),
      ui.h2("Keith Post"),
      ui.p(
        "For source code, click ", 
        ui.a("here", href="https://github.com/kpost34/imdb_ml_py"),
        
        ui.tags.br(),
        ui.tags.br(),
        
        "Check out my ", 
        ui.a("GitHub", href="https://github.com/kpost34"),
        " page for more coding projects",
        
        ui.tags.br(),
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
  @render.text
  def txt_sel():
    movie_title = input.txt_movie()
    
    if movie_title in df0['title'].values:
      return "Selected Movie"
    
    else:
      return ""
    
  @render.data_frame
  def tab_sel():
    movie_title = input.txt_movie()
    
    if not movie_title:
      return pd.DataFrame()
    
    elif movie_title not in df0['title'].values:
      return pd.DataFrame({'Message': [f"""The movie '{movie_title}' is not in the database. Please 
                            enter a valid movie title."""]})
    
    else: 
      df_sel = get_movie_info(df=df0, title=movie_title)
      
      return df_sel
  
  @render.text
  def txt_recs():
    movie_title = input.txt_movie()
    
    if movie_title in df0['title'].values:
      return "Recommended Movies"
    
    else:
      return ""
    
  @render.data_frame
  def tab_recs():
    movie_title = input.txt_movie()
    n_recs = input.sld_n_recs()
    sim_score = input.chk_sim_score()
    imdb_rank = input.chk_imdb_rank()
    
    if movie_title in df0['title'].values:
      df_out = get_rec_cosine(df=df0, 
                              title=movie_title, 
                              mat=cosine_sim, 
                              n=n_recs,
                              s_score=sim_score,
                              i_rank=imdb_rank)
      return df_out
    
    else:
      return pd.DataFrame()


app = App(app_ui, server)


#Updates:
#1) rename the columns to say: "Movie Title", "Similarity Score", and "IMDB Rank"
#2) add a row number field so that it ranks the recommendations from 1 to x
#3) add functionality for checkboxes
#4) by default, it should return the title as well as the release year and genre (and maybe director
  #and star1?)
#5) populate instructions tab

#6) populate more info: diagnostics (once completed)
#8) deploy app

