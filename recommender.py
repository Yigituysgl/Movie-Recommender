import pandas as pd
import numpy as np
from collections import defaultdict
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))

    ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(
        os.path.join(base_path, "u.data"),
        sep='\t',
        names=ratings_cols,
        encoding='latin-1'
    )

    item_cols = ['item_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + [f'genre_{i}' for i in range(19)]
    movies = pd.read_csv(
        os.path.join(base_path, "u.item"),
        sep='|',
        names=item_cols,
        usecols=[0, 1],
        encoding='latin-1'
    )

    data = pd.merge(ratings, movies, on='item_id')
    return data, ratings, movies

def train_model(data):
    reader = Reader(rating_scale=(1, 5))
    data_surp = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)
    trainset, testset = train_test_split(data_surp, test_size=0.2)
    model = SVD()
    model.fit(trainset)
    return model, testset

def get_predictions(model, testset):
    return model.test(testset)


def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def evaluate(predictions, k=5, threshold=4.0):
    precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=threshold)
    print(f"RMSE: {accuracy.rmse(predictions, verbose=False):.4f}")
    print(f"Precision@{k}: {sum(precisions.values()) / len(precisions):.4f}")
    print(f"Recall@{k}: {sum(recalls.values()) / len(recalls):.4f}")

def precision_recall_at_k(predictions, k=5, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k])
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
    return precisions, recalls


def recommend_movies_for_user(user_id, top_n, movies):
    recommended_ids = [iid for (iid, _) in top_n[user_id]]
    return movies[movies['item_id'].isin(recommended_ids)][['movie_title']]


if __name__ == '__main__':
    data, ratings, movies = load_data()
    model, testset = train_model(data)
    predictions = get_predictions(model, testset)
    top_n = get_top_n(predictions, n=5)
    evaluate(predictions)
    print("\nRecommended Movies for User 1:")
    print(recommend_movies_for_user(1, top_n, movies))
