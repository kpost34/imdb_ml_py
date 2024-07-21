# Objects for app===================================================================================

str_instruct = """Please enter the title of a movie in the dialog box. Use the slider to set the number of
          recommended movies returned. The app will always return the following information about the
          movie entered into the box: title, year released, genre(s), director, and first star. By default, 
          the app will return the same information for the recommended movies as well as the recommendation
          rank or 'rec rank' (i.e., strongest to weakest recommendation out of the total list). The 
          similarity score (i.e., from the cosine similarity matrix using engineered features) and
          IMDB Rank (within the top 1000) are optionally returned using the checkboxes."""
          
str_data_prep1 = """ were used as the data source. This data file contains the following information: movie title,
          year of release, certificate (rating), runtime, genre, IMDB rating, overview, meta-score,
          director, stars 1-4, and money grossed."""
          
str_data_prep2 = """The movie genre was converted into dummy variables as one or more genres can apply to a single
          movie. A term frequency-inverse document frequency (TF-IDF) was performed on the movie overview, 
          removing all stop words, and retaining the top 11 words (TF-IDF > 0.01). These steps were performed prior 
          to imputation as genre was a 'composite' variable and overview was unique text."""
          
str_data_prep3 = """The data were assessed for missingness, which was found for certificate, meta-score, and
          gross money earned. Little's missing completely at random (MCAR) test was performed, which was 
          significant suggesting that missingness was not completely at random. Certificate was imputed using the most
          frequent category. Meta-score and gross money earned were imputed using K-nearest neighbors."""
          
str_data_prep4 = """Multicollinearity was assessed for all numerical fields and no pair was found to be highly
          correlated (Spearman rank correlation < 0.9 for all comparisons). Rare label encoding was conducted on certificate,
          director, and all four star fields by retaining the most common categories and binning the remainder
          as "Other". These six features underwent one-hot encoding to generate sets of dummy variables.
          All remaining numerical fields (e.g., excluding 0-1 dummy variables, strings, etc.) underwent
          min-max scaling because they had non-normal distributions."""

str_algo = """A cosine similarity matrix was generated using the cleaned, feature-engineered top 1000
        IMDB movies dataset. A custom function called get_rec_cosine() was developed, which takes in
        the movie title (and year, if necessary) to generate a dataframe of recommended movies:
        titles, years released, genre(s), directors, first stars, similarity scores, and positions in 
        the IMDB top 1000."""

str_diagnostics1 = """Precision at k (P@k) was used to evaluate the recommendation algorithm. This metric
        determines the proportions of the top-k algorithm-recommended movies that have high ratings from the 
        same users who rated the input movie highly. Out of the 1000 movies in the top IMDB movie list, 
        718 were used in calculating P@k with user ratings data from """
        
str_diagnostics2 = """Because users can rate the input movie highly and have not watched (and thus rated) the
        top-k recommended movies, it's possible to calculate P@k on a per-movie basis or a 
        per-movie-instance basis. The former means that the proportion of top-k recommended movies
        rated highly by the group of users who rated the inputted movie highly would be the score 
        that becomes one of the potentially 718 numbers averaged to determine the P@k. The latter
        holds onto the actual fraction (number of top-k recommended movies rated highly over the
        total number of top-k recommended movies watched by the group of users who rated the inputted
        movie highly) when computing the P@k."""
        
str_diagnostics3 = """The P@k was computed using a threshold rating of at least 4/5 for 5, 10, and 20 recommended 
        movies and using both per-movie and per-movie-instance bases:"""

str_diagnostics4 = """P@k greater than 0.5 is considered a strong performance, which is accomplished
          when k = 5, 10, or 20 and on both per-movie and per-movie-instance bases."""







