import streamlit as st
import pandas as pd
import numpy as np
import ast

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ==============================
# LOAD DATA
# ==============================
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

df = movies.merge(credits, on="title")

df = df[['title','genres','budget','production_companies',
         'production_countries','release_date',
         'cast','crew','vote_average']]

df.rename(columns={'vote_average':'rating'}, inplace=True)

# ==============================
# CLEAN FUNCTIONS
# ==============================
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def get_names(x):
    return [i['name'] for i in safe_eval(x)]

def get_first(x):
    L = get_names(x)
    return L[0] if L else "unknown"

def get_actor(x):
    L = get_names(x)
    return L[0] if L else "unknown"

def get_director(x):
    for i in safe_eval(x):
        if i.get('job') == 'Director':
            return i.get('name','unknown')
    return "unknown"

# ==============================
# FEATURE ENGINEERING
# ==============================
df['genre_list'] = df['genres'].apply(get_names)
df['studio'] = df['production_companies'].apply(get_first)
df['country'] = df['production_countries'].apply(get_first)
df['actor'] = df['cast'].apply(get_actor)
df['director'] = df['crew'].apply(get_director)

df = df[df['genre_list'].apply(lambda x: 'Animation' in x)]
df = df[(df['budget'] > 0) & (df['rating'] > 0)]

df['budget'] = df['budget'] / 1000000

df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df.dropna(subset=['release_date'], inplace=True)

df['month'] = df['release_date'].dt.month

def season(m):
    if m in [5,6,7]: return "Summer"
    elif m in [11,12]: return "Holiday"
    else: return "Normal"

df['season'] = df['month'].apply(season)

def screen_category(b):
    if b >= 100: return "Wide"
    elif b >= 40: return "Medium"
    else: return "Limited"

df['screen'] = df['budget'].apply(screen_category)

def classify(r):
    if r >= 6.5: return "Hit"
    elif r >= 5.5: return "Average"
    else: return "Flop"

df['status'] = df['rating'].apply(classify)

# ==============================
# EXTRA LIST
# ==============================
extra_directors = ["Makoto Shinkai","Hayao Miyazaki","Satoshi Kon","Mamoru Hosoda","Tim Burton"]
extra_actors = ["Ryunosuke Kamiki","Megumi Hayashibara","Tom Hanks","Scarlett Johansson","Chris Pratt"]

# ==============================
# ENCODING
# ==============================
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_list'])

actor_dummies = pd.get_dummies(df['actor'])
director_dummies = pd.get_dummies(df['director'])
country_dummies = pd.get_dummies(df['country'])
studio_dummies = pd.get_dummies(df['studio'])
season_dummies = pd.get_dummies(df['season'])
screen_dummies = pd.get_dummies(df['screen'])

X = np.hstack((
    genre_encoded,
    actor_dummies.values,
    director_dummies.values,
    country_dummies.values,
    studio_dummies.values,
    season_dummies.values,
    screen_dummies.values,
    df[['budget']].values
))

y_class = df['status']
y_rating = df['rating']

# ==============================
# MODELS
# ==============================
clf = RandomForestClassifier(n_estimators=200, max_depth=15)
reg = RandomForestRegressor(n_estimators=200, max_depth=15)

clf.fit(X, y_class)
reg.fit(X, y_rating)

# ==============================
# UI
# ==============================
st.title("🎬 Movie Success Predictor")

genre_input = st.multiselect("🎭 Genre", mlb.classes_)

# 🔥 ALL LIST (NO FILTER)
all_actors = sorted(list(df['actor'].unique()) + extra_actors)
all_directors = sorted(list(df['director'].unique()) + extra_directors)

actor_input = st.selectbox("🎤 Actor", all_actors)
director_input = st.selectbox("🎬 Director", all_directors)
country_input = st.selectbox("🌍 Country", sorted(df['country'].unique()))
studio_input = st.selectbox("🏢 Studio", sorted(df['studio'].unique()))

season_input = st.selectbox("📅 Season", ["Summer","Holiday","Normal"])
screen_input = st.selectbox("🎬 Screen Reach", ["Limited","Medium","Wide"])

budget = st.slider("💰 Budget (Million USD)", 1, 300, 50)

# ==============================
# PREDICT
# ==============================
if st.button("Predict"):

    def encode(value, columns):
        vec = np.zeros((1, len(columns)))
        if value in columns:
            vec[0][list(columns).index(value)] = 1
        return vec

    genre_vec = mlb.transform([genre_input])
    actor_vec = encode(actor_input, actor_dummies.columns)
    director_vec = encode(director_input, director_dummies.columns)
    country_vec = encode(country_input, country_dummies.columns)
    studio_vec = encode(studio_input, studio_dummies.columns)
    season_vec = encode(season_input, season_dummies.columns)
    screen_vec = encode(screen_input, screen_dummies.columns)

    final_input = np.hstack((
        genre_vec,
        actor_vec,
        director_vec,
        country_vec,
        studio_vec,
        season_vec,
        screen_vec,
        [[budget]]
    ))

    probs = clf.predict_proba(final_input)[0]
    prediction = clf.classes_[np.argmax(probs)]

    rating_pred = reg.predict(final_input)[0]

    st.success(f"🎯 Prediction: {prediction}")
    st.write(f"⭐ Expected Rating: {round(rating_pred,2)}")

    st.subheader("📊 Confidence")
    for i, cls in enumerate(clf.classes_):
        st.write(f"{cls}: {round(probs[i]*100,2)}%")