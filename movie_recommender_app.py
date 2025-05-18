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
genre_columns = [col for col in movies.columns if col.startswith('genre_')]

unique_users = sorted(data['user_id'].unique())


st.sidebar.title(" Choose ")
user_id = st.sidebar.selectbox("Select User ID:", unique_users)


movies['release_year'] = movies['movie_title'].str.extract(r'\((\d{4})\)').astype(float)
movies = movies.dropna(subset=['release_year'])
movies['release_year'] = movies['release_year'].astype(int)

year_min, year_max = int(movies['release_year'].min()), int(movies['release_year'].max())
year_range = st.sidebar.slider("Filter by Release Year", year_min, year_max, (year_min, year_max))



genre_path = 'ml-100k/u.genre'
genre_dict = {}
with open(genre_path, 'r') as f:
    for line in f:
        name, idx = line.strip().split('|')
        genre_dict[int(idx)] = name


item_path = 'ml-100k/u.item'

genre_columns = [f"genre_{genre_dict[i]}" for i in range(len(genre_dict)) if i in genre_dict]
movies = pd.read_csv(item_path, sep='|', encoding='latin-1', header=None,
                     usecols=[1] + list(range(5, 5 + len(genre_dict))),
                     names=['movie_title'] + genre_columns)


selected_genres = st.sidebar.multiselect("Filter by Genre:", genre_columns)




if st.sidebar.button("🎬 Recommend Movies"):
    with st.spinner("Training model and fetching recommendations..."):
        model, testset = train_model(data)
        predictions = get_predictions(model, testset)
        top_n = get_top_n(predictions, n=10)
        recommended_df = recommend_movies_for_user(user_id, top_n, movies)

       
        genre_cols = [col for col in movies.columns if col.startswith("genre_")]
        recommended_df = pd.merge(recommended_df, movies[['movie_title', 'release_year'] + genre_cols], on='movie_title', how='left')

       
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


