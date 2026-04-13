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

df = df[['title','genres','budget','production_countries',
         'cast','crew','vote_average']]

df.rename(columns={'vote_average':'rating'}, inplace=True)

# ==============================
# FUNCTIONS
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
df['country'] = df['production_countries'].apply(get_first)
df['actor'] = df['cast'].apply(get_actor)
df['director'] = df['crew'].apply(get_director)

df = df[df['genre_list'].apply(lambda x: 'Animation' in x)]
df = df[df['rating'] > 0]

df['budget'] = df['budget'] / 1000000

# ==============================
# ENCODING
# ==============================
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_list'])

actor_dummies = pd.get_dummies(df['actor'])
director_dummies = pd.get_dummies(df['director'])
country_dummies = pd.get_dummies(df['country'])

X = np.hstack((
    genre_encoded,
    actor_dummies.values,
    director_dummies.values,
    country_dummies.values,
    df[['budget']].values
))

# ==============================
# MODELS
# ==============================
clf = RandomForestClassifier(n_estimators=200, max_depth=15)
reg = RandomForestRegressor(n_estimators=200, max_depth=15)

clf.fit(X, (df['rating'] > 6).astype(int))
reg.fit(X, df['rating'])

# ==============================
# UI
# ==============================
st.title("🎬 Movie Success Predictor ")

genre_input = st.multiselect(" Genre", mlb.classes_)
actor_input = st.selectbox(" Actor", sorted(df['actor'].unique()))
director_input = st.selectbox(" Director", sorted(df['director'].unique()))
country_input = st.selectbox(" Country", sorted(df['country'].unique()))

budget = st.slider(" Budget (Million USD)", 1, 300, 50)

screen_count = st.number_input(" Number of Screens", min_value=50, max_value=20000, value=800)
run_weeks = st.slider(" Weeks in Theater", 1, 20, 4)
screen_stability = st.selectbox(" Screen Stability", ["Drop Fast","Stable","Growing"])

ticket_price = st.slider(" Ticket Price ($)", 2, 20, 10)
shows_per_day = st.slider(" Shows per Day", 1, 15, 8)

# ==============================
# PREDICT
# ==============================
if st.button("Predict"):

    def encode(value, columns):
        vec = np.zeros((1, len(columns)))
        if value in columns:
            vec[0][list(columns).index(value)] = 1
        return vec

    # Encoding
    genre_vec = mlb.transform([genre_input])
    actor_vec = encode(actor_input, actor_dummies.columns)
    director_vec = encode(director_input, director_dummies.columns)
    country_vec = encode(country_input, country_dummies.columns)

    final_input = np.hstack((
        genre_vec,
        actor_vec,
        director_vec,
        country_vec,
        [[budget]]
    ))

    # ML
    probs = clf.predict_proba(final_input)[0]
    rating_pred = reg.predict(final_input)[0]

    # ==============================
    #  BASE REVENUE
    # ==============================
    days = run_weeks * 7
    revenue = screen_count * shows_per_day * ticket_price * days / 1000000

    # ==============================
    #  STABILITY EFFECT (FIXED)
    # ==============================
    if screen_stability == "Drop Fast":
        revenue *= 0.6
    elif screen_stability == "Stable":
        revenue *= 1.0
    elif screen_stability == "Growing":
        revenue *= 1.5

    # ==============================
    # PROFIT
    # ==============================
    profit = revenue - budget

    # ==============================
    # DECISION
    # ==============================
    if profit < 0:
        prediction = "Flop"
    elif abs(profit) <= 5:
        prediction = "Average"
    else:
        prediction = "Hit"

    # ==============================
    # COLOR OUTPUT
    # ==============================
    if prediction == "Hit":
        st.success(f" Prediction: {prediction}")
    elif prediction == "Average":
        st.warning(f" Prediction: {prediction}")
    else:
        st.error(f" Prediction: {prediction}")

    st.write(f" Expected Rating: {round(rating_pred,2)}")

    # ==============================
    # BOX OFFICE
    # ==============================
    st.subheader(" Box Office")

    st.write(f"Revenue: ${round(revenue,2)} Million")

    if profit >= 0:
        st.write(f"Profit: ${round(profit,2)} Million")
    else:
        st.write(f"Loss: ${round(abs(profit),2)} Million")

    # ==============================
    # CONFIDENCE
    # ==============================
    st.subheader(" Confidence")
    st.write(f"Hit Chance: {round(probs[1]*100,2)}%")
    st.write(f"Flop Chance: {round(probs[0]*100,2)}%")

    # ==============================
    # SMART WHY
    # ==============================
    st.subheader(" Why this prediction?")

    if profit < 0:
        st.write(" The movie could not recover its budget.")
    else:
        st.write(" The movie earned more than its budget.")

    if screen_stability == "Drop Fast":
        st.write(" Audience dropped quickly, reducing revenue.")

    elif screen_stability == "Growing":
        st.write(" Positive word of mouth increased revenue.")

    if screen_count > 1500:
        st.write(" Wide release increased reach.")

    if run_weeks >= 8:
        st.write(" Long run boosted earnings.")