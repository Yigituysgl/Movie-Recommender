# Movie-Recommender

A personal movie recommender built on the MovieLens 100K dataset. It learns from past ratings using SVD collaborative filtering and suggests movies a user hasn't rated yet. The recommendations are shown in a small Streamlit web app where you can filter by release year and genre.

## What it does

- Trains an SVD model (from the [Surprise](https://surpriselib.com/) library) on user–movie ratings.
- Predicts a rating for every movie a user hasn't rated, and recommends the highest-scoring ones.
- Evaluates the model separately on a held-out test split (RMSE, Precision@k, Recall@k).
- Provides a Streamlit app: choose a user ID, optionally filter by release year and genre, and get the top 10 recommendations.

## Dataset

[MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) from GroupLens Research, included in this repository:

| File | Contents |
|------|----------|
| `u.data` | Ratings, tab-separated: `user_id`, `item_id`, `rating`, `timestamp` |
| `u.item` | Movies, `\|`-separated: ID, title, release date, IMDb URL and 19 genre flags |

Figures computed from the files:

- 100,000 ratings from 943 users on 1,682 movies
- Ratings are whole numbers from 1 to 5; the mean rating is 3.53
- Every user has rated at least 20 movies

A few titles appear twice under different item IDs (for example *Nightwatch (1997)* is items 1477 and 1625), so the same title can show up twice in a recommendation list.

## Method: SVD collaborative filtering

Collaborative filtering predicts a user's taste from the rating patterns of all users, without using the movies' content. This project uses Surprise's `SVD`, a matrix factorization model. Each user and each movie is represented by a vector of latent factors, and a rating is predicted as:

```
predicted rating = global mean + user bias + movie bias + (user factors · movie factors)
```

The biases capture that some users rate generously and some movies are rated highly by everyone. The dot product captures how well a user's tastes match a movie's characteristics. The factors are learned with stochastic gradient descent. The code uses Surprise's defaults: 100 factors, 20 epochs, learning rate 0.005, regularization 0.02.

To produce recommendations (`train_full_model` and `get_unrated_predictions` in `recommender.py`):

1. Train SVD on **all** ratings.
2. For the chosen user, predict a rating for every movie they haven't rated.
3. Sort by predicted rating. The app then applies the year and genre filters and shows the top 10.

## Evaluation

Evaluation is kept separate from recommendation and uses a random 80/20 split (`train_model` in `recommender.py`). The model is trained on 80% of the ratings and scored on the held-out 20%:

- **RMSE**: the root mean squared error between predicted and actual ratings on the test set. Lower is better.
- **Precision@5 / Recall@5**: a rating of 4 or higher counts as "relevant". For each user, the test-set movies are sorted by predicted rating. Among the top 5, a movie counts as "recommended" if its predicted rating is 4 or higher.
  - Precision@5 = relevant and recommended ÷ recommended
  - Recall@5 = relevant and recommended ÷ all relevant test movies for that user
  - Both are averaged over users. A user with no recommended (or no relevant) movies scores 0.

### Results

Output of `python recommender.py`. The split and SVD's starting values are random and no seed is set, so scores vary slightly between runs:

| Run | RMSE | Precision@5 | Recall@5 |
|-----|------|-------------|----------|
| 1 | 0.9365 | 0.6457 | 0.2470 |
| 2 | 0.9372 | 0.6359 | 0.2312 |
| 3 | 0.9376 | 0.6362 | 0.2340 |

On the 1–5 scale, the typical prediction error is about 0.94 stars (RMSE, which weights large errors more heavily). Per user, on average about 64% of the top-5 movies predicted at 4 or higher were ones the user actually rated 4 or higher.

These runs used Python 3.11.16, scikit-surprise 1.1.4, pandas 2.3.3, numpy 1.24.4 and streamlit 1.64.0.

## Setup on Windows (conda)

`scikit-surprise` is a C extension, and `pip` often has no prebuilt Windows wheel for it. Installing it with pip then needs Microsoft C++ Build Tools. conda-forge provides a prebuilt package, so the steps below avoid that.

1. **Install conda.** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) if you don't already have one.

2. **Open "Anaconda Prompt"** from the Start menu. It has `conda` on its PATH, which a normal terminal may not.

3. **Get the code:**

   ```bash
   git clone https://github.com/Yigituysgl/Movie-Recommender.git
   cd Movie-Recommender
   ```

4. **Create the environment.** Use Python 3.11: the pinned numpy 1.24 has no builds for Python 3.12. This downloads roughly 300 MB.

   ```bash
   conda create -n movierec python=3.11 numpy=1.24 pandas scikit-learn streamlit scikit-surprise -c conda-forge
   ```

5. **Activate it:**

   ```bash
   conda activate movierec
   ```

6. **Run the tests:**

   ```bash
   python test_recommender.py
   ```

7. **Print the evaluation metrics and sample recommendations for user 1:**

   ```bash
   python recommender.py
   ```

8. **Start the web app:**

   ```bash
   streamlit run movie_recommender_app.py
   ```

   It opens at http://localhost:8501. Choose a user ID in the sidebar, optionally set a year range and genres, and click **🎥 Recommend Movies**. The first click trains the model, which takes a few seconds; later clicks reuse it. Stop the app with `Ctrl+C`.

Without conda, `pip install -r requirement.txt` with Python 3.11 should also work, but `scikit-surprise` will likely need Microsoft C++ Build Tools ("Desktop development with C++") to compile. This route hasn't been tested.

## Project files

| File | Purpose |
|------|---------|
| `recommender.py` | Data loading, model training, predictions, evaluation |
| `movie_recommender_app.py` | Streamlit web app |
| `test_recommender.py` | Tests (run with `python test_recommender.py`) |
| `Personal_Recom_System.ipynb` | Notebook used during development |
| `requirement.txt` | pip dependencies |
| `svd_model.pkl` | Pickled file, probably a saved model; not loaded by any code |

## Limitations

- Only existing MovieLens users (IDs 1–943) can get recommendations; there is no way to add your own ratings.
- No random seed is set, so metrics and recommendations change slightly between runs and app restarts.
- Duplicate titles in the dataset can appear twice in a recommendation list.

## Acknowledgements

MovieLens data: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5(4). https://doi.org/10.1145/2827872
