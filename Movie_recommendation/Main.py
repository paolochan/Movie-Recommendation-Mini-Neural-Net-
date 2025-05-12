#!/usr/bin/env python
# coding: utf-8

# In[20]:


import sklearn
print(sklearn.__version__)


# In[21]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam





# In[22]:


## Part one: Gathering data

#movies
df_movies = pd.read_csv("imdb_top_1000.csv")  # or whatever your filename is
df_movies["Series_Title"].head()

#Synthetic user data
np.random.seed(12)
user_ratings = {title: np.round(np.random.uniform(0.0, 1.0), 2) for title in df_movies['Series_Title']}

# Assign to DataFrame
df_movies['User_Rating'] = df_movies['Series_Title'].map(user_ratings)


# In[23]:


## Step 2: preprocessing data

## Cleaning 
df_movies= df_movies.drop(["Gross", "Poster_Link", "Certificate", "No_of_Votes","Meta_score","Runtime"], axis=1) ## Drop data we don't want to use
df_movies["Series_Title"]= df_movies["Series_Title"].str.lower().str.strip() 
df_movies["Genre"]= df_movies["Genre"].str.lower().str.strip()
df_movies['Genre_List'] = df_movies['Genre'].str.split(', ') #splitting genres into a list instead of string

## Merge the actors into one list 
df_movies["Stars"] = df_movies.apply(
    lambda row: [
        str(row["Star1"]).lower().strip(),
        str(row["Star2"]).lower().strip(),
        str(row["Star3"]).lower().strip(),
        str(row["Star4"]).lower().strip()
    ], axis=1
)
actor_counts = Counter(actor for sublist in df_movies['Stars'] for actor in sublist) # count actor instances

top_actors = [actor for actor, count in actor_counts.items() if count >= 3]  # top actors appear more than 5 times
# Keep only top actors in each movie
def filter_top_actors(stars_list):
    return [actor for actor in stars_list if actor in top_actors]
df_movies['Stars'] = df_movies['Stars'].apply(filter_top_actors)

#Normalize IMDB rating 
scaler = MinMaxScaler()
df_movies['Nrating']= scaler.fit_transform(df_movies[['IMDB_Rating']])


#Encoding 

#encoding genres
mlb = MultiLabelBinarizer()
##mlb.fit_transform([df_movies['Genre_List']])
genre_encoded = pd.DataFrame(
    mlb.fit_transform(df_movies['Genre_List']), columns=mlb.classes_, index=df_movies['Series_Title'])

#Encoding Director
top_25_directors = df_movies['Director'].value_counts().nlargest(25).index
df_movies['Director'] = df_movies['Director'].where(df_movies['Director'].isin(top_25_directors), 'Other')
ohe = OneHotEncoder(sparse_output=False)
director_encoded= pd.DataFrame(ohe.fit_transform(df_movies[['Director']]),columns=ohe.get_feature_names_out(['Director']),index=df_movies['Series_Title'])

#Encoding actors
mlba = MultiLabelBinarizer()
actor_encoded = pd.DataFrame(mlba.fit_transform(df_movies['Stars']), columns= mlba.classes_, index= df_movies['Series_Title'])



#Tokenizing and Embedding of overview
df_movies['tokens'] = df_movies['Overview'].fillna('').apply(word_tokenize)

w2v_model = Word2Vec(sentences=df_movies["tokens"], vector_size=100, window=5, min_count=2, workers=4)
def average_vector(tokens) :
    vectors = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)
df_movies['overview_vector'] = df_movies['tokens'].apply(average_vector)

X = np.hstack([
    genre_encoded.values,
    actor_encoded.values,
    director_encoded.values,
    np.vstack(df_movies['overview_vector']),
    df_movies[['Nrating']].values
])
y=df_movies['User_Rating'].values





# In[24]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)


# In[27]:


#Using Keras to make a neural network moddel

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  
])
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',   
              metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=20,              # Number of full passes over the training data
    batch_size=32,          # Number of samples per gradient update
    validation_split=0.1,   # Use 10% of training data for validation
    verbose=1  )             # Print progress bar

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


# In[43]:


valid_genres=[]
def newInfo():
    print("Please enter details on a new movie:")
    title =input("Title: ").strip().lower()

    #getting available genres
    valid_genres = [
    'action', 'adventure', 'animation', 'biography', 'comedy', 'crime',
    'drama', 'family', 'fantasy', 'film-noir', 'history', 'horror',
    'music', 'musical', 'mystery', 'romance', 'sci-fi', 'sport',
    'thriller', 'war', 'western'
]
    while True:
        print(f"Available genres:\n{', '.join(valid_genres)}")
        genres = input("Enter genres (comma-separated): ").strip().lower().split(',')
        genres = [g.strip() for g in genres]
        
        if all(g in valid_genres for g in genres):
            break
        else:
            print("❌ One or more genres are invalid. Please choose only from the listed genres.\n")

    actors = input("Main actors (comma-separated): ").strip().lower().split(',')
    actors = [a.strip() for a in actors]
    director = input("Director: ").strip().lower()
    imdb_rating = float(input("IMDB rating (0.0–10.0): "))
    overview = input("Overview: ").strip().lower()
    return {
        "Series_Title": title,
        "Genre_List": genres,
        "Stars": actors,
        "Director": director,
        "IMDB_Rating": imdb_rating,
        "Overview":overview
    
    }

    


# In[45]:





# In[50]:


def preprocess_new_movie(movie, w2v_model, genre_encoder, actor_encoder, director_encoder, scaler):
    genre_vec = genre_encoder.transform([movie["Genre_List"]])

    top_actors = actor_encoder.classes_
    movie_actors = [actor for actor in movie["Stars"] if actor in top_actors]
    actor_vec = actor_encoder.transform([movie_actors])

    director = movie["Director"].strip()
    if director not in director_encoder.categories_[0]:
        director = "Other"
    director_vec = director_encoder.transform([[director]])

    tokens = word_tokenize(movie["Overview"])
    vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    overview_vec = np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)
    overview_vec = overview_vec.reshape(1, -1)

    rating_norm = scaler.transform([[movie["IMDB_Rating"]]])

    return np.hstack([genre_vec, actor_vec, director_vec, overview_vec, rating_norm])

# === Run the prediction ===
new_movie = newInfo()
X_new = preprocess_new_movie(new_movie, w2v_model, mlb, mlba, ohe, scaler)
score = model.predict(X_new.reshape(1, -1))[0][0]

print(f"\nPredicted score: {score:.2f}")
print("✅ User will likely like this movie!" if score > 0.6 else "❌ User probably won't like this movie.")


# In[ ]:




