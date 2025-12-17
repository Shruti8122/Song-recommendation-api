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

songs_data = pd.read_csv("spotify_millsongdata.csv")

songs_data["artist"] = songs_data["artist"].fillna("")
songs_data["text"] = songs_data["text"].fillna("")
songs_data["song"] = songs_data["song"].fillna("")

combined_features = songs_data["artist"] + " " + songs_data["text"]

vectorizer = TfidfVectorizer(stop_words="english")
feature_vectors = vectorizer.fit_transform(combined_features)

list_of_all_songs = songs_data["song"].tolist()

class SongRequest(BaseModel):
    song: str

@app.get("/")
def root():
    return {"message": "Song Recommendation API is running"}

@app.post("/recommendation")
def recommend(request: SongRequest):
    song_name = request.song

    close_matches = difflib.get_close_matches(
        song_name, list_of_all_songs, n=1
    )

    if not close_matches:
        return {"error": "Song not found"}

    matched_song = close_matches[0]

    song_index = songs_data[
        songs_data.song == matched_song
    ].index[0]

    similarity_scores = cosine_similarity(
        feature_vectors[song_index],
        feature_vectors
    )[0]

    similarity_list = list(enumerate(similarity_scores))

    sorted_songs = sorted(
        similarity_list, key=lambda x: x[1], reverse=True
    )

    recommendations = []
    for index, score in sorted_songs[1:21]:
        recommendations.append({
            "song": songs_data.iloc[index]["song"],
            "artist": songs_data.iloc[index]["artist"]
        })

    return {
        "matched_song": matched_song,
        "recommendations": recommendations
    }
