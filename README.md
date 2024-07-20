# **Movie Recommender Project**
A movie recommendation system was developed, evaluated, and showcased as a Shiny for Python app 

## Summary
The objective of this project was to develop a machine learning algorithm that could recommend a set of movies after a movie is provided using data from the [top 1000 movies by IMDB Rating](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows). A suite of variables describing the movies (i.e., year of release, certificate (rating), genre(s), IMDB rating, overview, meta-score, director, stars 1-4, and money grossed) underwent feature engineering (e.g., TF-IDF on overview, rare-label encoding on director, min-max scaling of numerical variables) and imputation of missing values prior to developing a cosine similarity matrix. A custom function (get_rec_cosine) was developed that could return a set of similar movies given a selected movie from the dataset.

Diagnostics were evaluated by measuring P@k for 5, 10, and 20 recommended movies using a threshold rating of 4/5 to determine whether a movie was considered 'rated highly'. User ratings were obtained from [Kaggle's The Movie Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). The P@5, P@10, and P@20 scores, whether evaluated on a per-movie or per-movie-instance basis, were at least 0.59 indicating that the recommendation system performs strongly.

A Shiny for Python App was developed so that users could test out the algorithm.


## App
+ [Shiny for Python App](need to add) 

#### **Project Creator: Keith Post**
+ [Github Profile](https://github.com/kpost34) 
+ [LinkedIN Profile](https://www.linkedin.com/in/keith-post/)
+ [Email](mailto:keithhpost@gmail.com)