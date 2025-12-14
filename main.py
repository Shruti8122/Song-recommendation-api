# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# from fastapi.middleware.cors import CORSMiddleware
# import difflib

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# movies_data = pd.read_csv("songs.csv")

# selected_features=['Title','Artist','Top Genre']
# for feature in selected_features:
#     movies_data[feature]=movies_data[feature].fillna('')

# combined_features=movies_data['Title']+movies_data['Artist']+movies_data['Top Genre']
# vectorizer = TfidfVectorizer()
# feature_vectors = vectorizer.fit_transform(combined_features)

# similarity = cosine_similarity(feature_vectors)

# list_of_all_titles = movies_data['Title'].to_list()

# class MovieRequest(BaseModel):
#     movie: str

# @app.get("/")
# def root():
#    return {"message": "Movie Recommendation API"}

# @app.post("/recommendation")
# def recommend(request: MovieRequest):
#     movie_name = request.movie
#     find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
#     close_match = find_close_match[0]

#     index_of_the_movie = movies_data[movies_data.title == close_match]['Index'].values[0]

#     similarity_score = list(enumerate(similarity[index_of_the_movie]))

#     sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

#     recommendations = []

#     for i, movie in enumerate(sorted_similar_movies[1:30]):
#        index= movie[0]
#        title = movies_data[movies_data.index == index]['Title'].values[0]
#        recommendations.append(title)

#     return {
#        "matched_movies": close_match,
#        "recommendations": recommendations
#     }
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import difflib

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the data
songs_data = pd.read_csv("songs.csv")

# Create the 'index' column if it doesn't exist (CRITICAL FIX from previous issue)
if 'index' not in songs_data.columns:
    songs_data['index'] = songs_data.index

# Feature engineering
selected_features = ['Title', 'Artist', 'Top Genre']
for feature in selected_features:
    songs_data[feature] = songs_data[feature].fillna('')

# Combine features
combined_features = songs_data['Title'] + ' ' + songs_data['Artist'] + ' ' + songs_data['Top Genre']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity
similarity = cosine_similarity(feature_vectors)

list_of_all_titles = songs_data['Title'].to_list()

# Pydantic model updated to SongRequest
class SongRequest(BaseModel):
    song: str

@app.get("/")
def root():
    return {"message": "Song Recommendation API"}

# Endpoint updated to /song_recommendation and variable names changed
@app.post("/song_recommendation")
def recommend_song(request: SongRequest):
    song_name = request.song
    
    # 1. Find the closest match to the input song name
    find_close_match = difflib.get_close_matches(song_name, list_of_all_titles)
    
    if not find_close_match:
        return {"error": f"Could not find a close match for '{song_name}' in the dataset."}
        
    close_match = find_close_match[0]

    # 2. Get the index of the matched song (using 'index' column)
    # NOTE: Changed .title to .Title and .Index to .index for standardisation and consistency with the added column
    index_of_the_song = songs_data[songs_data.Title == close_match]['index'].values[0]

    # 3. Get the similarity scores for that song
    similarity_score = list(enumerate(similarity[index_of_the_song]))

    # 4. Sort the scores
    sorted_similar_songs = sorted(similarity_score, key = lambda x:x[1], reverse = True)

    recommendations = []
    i = 0
    # 5. Extract top 30 similar songs (skipping the first one, which is the song itself)
    for index, score in sorted_similar_songs:
        if index == index_of_the_song:
            continue
            
        # Get the title using the index
        title = songs_data[songs_data['index'] == index]['Title'].values[0]
        recommendations.append(title)
        
        i += 1
        if i >= 30:
            break

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }