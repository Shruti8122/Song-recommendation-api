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

# Load the new dataset
songs_data = pd.read_csv("spotify_millsongdata.csv")

# Create the 'index' column if it doesn't exist
if 'index' not in songs_data.columns:
    songs_data['index'] = songs_data.index

# --- FEATURES UPDATED FOR NEW DATASET COLUMNS ('song', 'artist', 'text') ---
selected_features = ['song', 'artist', 'text']
for feature in selected_features:
    # Fill missing values with empty strings
    songs_data[feature] = songs_data[feature].fillna('')

# Combine features: Song Title + Artist Name + Lyrics Text for content analysis
combined_features = songs_data['song'] + ' ' + songs_data['artist'] + ' ' + songs_data['text']
vectorizer = TfidfVectorizer(stop_words='english')  # Added stop_words for better lyrics processing
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity
similarity = cosine_similarity(feature_vectors)

# Use the 'song' column for the list of titles
list_of_all_titles = songs_data['song'].to_list()
# --- END OF DATASET-SPECIFIC CHANGES ---

# Pydantic model
class SongRequest(BaseModel):
    song: str

@app.get("/")
def root():
    return {"message": "Song Recommendation API (Updated for spotify_millsongdata)"}

@app.post("/song_recommendation")
def recommend_song(request: SongRequest):
    song_name = request.song
    
    # 1. Find the closest match to the input song name
    find_close_match = difflib.get_close_matches(song_name, list_of_all_titles)
    
    if not find_close_match:
        return {"error": f"Could not find a close match for '{song_name}' in the dataset."}
        
    close_match = find_close_match[0]

    # 2. Get the index of the matched song (using 'index' column)
    # Match against the 'song' column
    index_of_the_song = songs_data[songs_data.song == close_match]['index'].values[0]

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
            
        # Get the title and artist using the index
        title = songs_data[songs_data['index'] == index]['song'].values[0]
        artist = songs_data[songs_data['index'] == index]['artist'].values[0] # Fetch artist as well
        
        if i < 30:
            # Append as "Title (by Artist)" for clearer output
            recommendations.append(f"{title} (by {artist})")
            i += 1
        else:
            break

    return {
        "matched_song": close_match,
        "recommendations": recommendations
    }