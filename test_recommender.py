from recommender import load_data, train_model, get_predictions, get_top_n, recommend_movies_for_user

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

if __name__ == '__main__':
    test_load_data()
    test_model_training()
    test_predictions()
