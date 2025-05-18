import streamlit as st
import pandas as pd
from recommender import load_data, train_model, get_predictions, get_top_n, recommend_movies_for_user

st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Personal Movie Recommender System")
st.markdown("Recommend movies based on your ratings using collaborative filtering.")

@st.cache_data
def get_loaded():
    return load_data()

data, ratings, movies = get_loaded()


genre_labels = [
    "Unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
genre_columns = [f"genre_{g}" for g in genre_labels]


item_path = "u.item"
movies = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None,
                     usecols=[0, 1, 2] + list(range(5, 24)),
                     names=["item_id", "movie_title", "release_date"] + genre_columns)


movies['release_year'] = pd.to_datetime(movies['release_date'], errors='coerce').dt.year
movies = movies.dropna(subset=['release_year'])
movies['release_year'] = movies['release_year'].astype(int)


user_id = st.sidebar.selectbox("Select User ID:", sorted(data['user_id'].unique()))
year_min, year_max = movies['release_year'].min(), movies['release_year'].max()
year_range = st.sidebar.slider("Filter by Release Year", year_min, year_max, (year_min, year_max))
selected_genres = st.sidebar.multiselect("Filter by Genre:", genre_columns)


if st.sidebar.button("🎥 Recommend Movies"):
    with st.spinner("Training model and fetching recommendations..."):
        model, testset = train_model(data)
        predictions = get_predictions(model, testset)
        top_n = get_top_n(predictions, n=10)
        recommended_df = recommend_movies_for_user(user_id, top_n, movies)

        
        recommended_df = recommended_df.dropna(subset=['release_year'])
        recommended_df['release_year'] = recommended_df['release_year'].astype(int)
        filtered_df = recommended_df[
            (recommended_df['release_year'] >= year_range[0]) &
            (recommended_df['release_year'] <= year_range[1])
        ]

        
        if selected_genres:
            for genre in selected_genres:
                filtered_df = filtered_df[filtered_df[genre] == 1]

        
        st.success(f"Here are your movie recommendations, User {user_id} 🎉")
        for title in filtered_df['movie_title'].values:
            st.write(f"✅ {title}")
else:
    st.info("Select options and click '🎥 Recommend Movies' to see results.")



    
