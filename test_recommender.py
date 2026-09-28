from recommender import (load_data, train_model, train_full_model, get_predictions,
                         get_unrated_predictions, get_top_n, recommend_movies_for_user,
                         compare_algorithms)

def test_load_data():
    data, ratings, movies = load_data()
    assert not data.empty, "Data is empty!"
    assert 'movie_title' in data.columns, "movie_title not in data!"
    assert 'user_id' in data.columns, "user_id missing!"
    print("✅ load_data test passed")

def test_model_training():
    data, _, _ = load_data()
    model, testset = train_model(data)
    assert model is not None, "Model is None!"
    assert len(testset) > 0, "Test set is empty!"
    print("✅ train_model test passed")

def test_predictions():
    data, _, movies = load_data()
    model, testset = train_model(data)
    predictions = get_predictions(model, testset)
    assert len(predictions) > 0, "No predictions made!"
    top_n = get_top_n(predictions, n=5)
    assert 1 in top_n, "User 1 not in top_n"
    recommended = recommend_movies_for_user(1, top_n, movies)
    assert not recommended.empty, "No recommendations for user 1"
    print("✅ predictions & recommend_movies test passed")

def test_unrated_recommendations():
    data, _, movies = load_data()
    model, trainset = train_full_model(data)
    predictions = get_unrated_predictions(model, trainset, 1)
    rated = set(data[data['user_id'] == 1]['item_id'])
    predicted = {iid for (_, iid, _, _, _) in predictions}
    assert predicted, "No unrated predictions for user 1"
    assert not predicted & rated, "Predicted movies user 1 already rated!"
    assert predicted | rated == set(data['item_id']), "Some unrated movies were skipped!"
    top_n = get_top_n(predictions, n=5)
    recommended = recommend_movies_for_user(1, top_n, movies)
    assert len(recommended) == 5, "Expected 5 recommendations for user 1"
    print("✅ unrated recommendations test passed")

def test_reproducible():
    data, _, _ = load_data()
    model_a, testset_a = train_model(data)
    model_b, testset_b = train_model(data)
    assert testset_a == testset_b, "Test split differs between runs!"
    assert get_predictions(model_a, testset_a) == get_predictions(model_b, testset_b), "Predictions differ between runs!"
    print("✅ reproducibility test passed")

def test_compare_algorithms():
    data, _, _ = load_data()
    results = compare_algorithms(data)
    assert set(results) == {'NormalPredictor', 'BaselineOnly', 'SVD'}, "Missing algorithms!"
    assert results['SVD'][0] < results['NormalPredictor'][0], "SVD RMSE not better than random!"
    print("✅ compare_algorithms test passed")

if __name__ == '__main__':
    test_load_data()
    test_model_training()
    test_predictions()
    test_unrated_recommendations()
    test_reproducible()
    test_compare_algorithms()
