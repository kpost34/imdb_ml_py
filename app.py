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


## String objects
import app_obj 


## Data
#clean data
root = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root, 'data')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

os.chdir(data_dir)
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
    
    ## Instructions tab
    ui.nav_panel("Instructions", 
      ui.div(
        ui.tags.br(),
        ui.p(app_obj.str_instruct),
        style="margin-left: 5%; margin-right: 5%; margin-bottom: 5%")
      ),
  
    ## More information tab
    ui.nav_panel("More info", 
      ui.div(
        ui.tags.br(),
        #create a collapseable tab
        ui.accordion(
        
        #data preparation section
        ui.accordion_panel("Data Preparation",
        
          ui.p(
            "The ", 
            ui.a("top 1000 movies by IMDB rating", 
            href="https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows",
            target="_blank"),
            app_obj.str_data_prep1
          ),
          
          ui.p(app_obj.str_data_prep2),
          ui.p(app_obj.str_data_prep3),
          ui.p(app_obj.str_data_prep4),
          open=False
        ),
        
        #algorithm section
        ui.accordion_panel("Algorithm",
        
          ui.p(app_obj.str_algo),
          open=False
        ),
        
        # ui.tags.br(),
        
        #diagnostics section
        ui.accordion_panel("Diagnostics",
        
          ui.p(
            app_obj.str_diagnostics1,
            ui.a("Kaggle's The Movies Dataset.",
            href="https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset",
            target="_blank")
          ),
          ui.p(app_obj.str_diagnostics2),
          ui.p(app_obj.str_diagnostics3),
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
          ui.p(app_obj.str_diagnostics4),
          open=False
        )
      ),
      
      style="margin-left: 5%; margin-right: 5%; margin-bottom: 5%")
    ),
    
    #developer info tab
    ui.nav_panel("Developer", 
      ui.div(
        ui.tags.br(),
        ui.h2("Keith Post"),
        ui.p(
          "For source code, click ", 
          ui.a("here", href="https://github.com/kpost34/imdb_ml_py", target="_blank"),
          
          ui.tags.br(),
          ui.tags.br(),
          
          "Check out my ", 
          ui.a("GitHub", href="https://github.com/kpost34", target="_blank"),
          " page for more coding projects",
          
          ui.tags.br(),
          ui.tags.br(),
          
          ui.a("LinkedIn", href="https://www.linkedin.com/in/keith-post/", target="_blank")
        ),
        style="margin-left: 5%; margin-right: 5%; margin-bottom: 5%")
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



#TO DO:
#1) Get hyperlinks to open in new tabs
#2) Fix hyperlinks
#3) Try to deploy app while it imports functions from app_fns.py
#4) Add horizontal margins to text
#5) Add vertical margins to text
#6) Create separate obj script which gets imported and try to deploy app using that architecture
#9) Adjust widths of table columns
#10) Set up "More Info" page such that each header is an accordion


#7) Use external script to create merged DF
#8) Read in merged DF from item 7












