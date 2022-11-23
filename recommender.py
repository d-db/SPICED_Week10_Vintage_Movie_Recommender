import numpy as np
import pandas as pd
import pickle
import random

from sklearn.decomposition import NMF

# ignore NMF warnings
import warnings
warnings.filterwarnings("ignore")


def ask_user_recommendations():

    with open('./data/more_than_100.bin', 'rb') as f:
        more_than_100 = pickle.load(f)

    counter = 0
    user_ratings = {}
    films = []

    while counter < 5:
        film = random.choice(more_than_100.keys())
        if film not in films:
            films.append(film)
            user_input = input(f"What is your rating (0-5) of {film} (Press 'q' if you haven't seen it): ")
            if user_input == "q":
                pass
            else:
                user_ratings[film] = int(user_input)
            counter += 1

    return user_ratings


def recommend_nmf(user_ratings, model="nmf.sav", k=10):
    """Filters and recommends the top k movies
    for any given input query based
    on a trained NMF model.

    Parameters
    ----------
    query : dict
        A dictionary of movies already seen.
        Takes the form {"movie_A": 3, "movie_B": 3} etc
    model : pickle
        pickle model read from disk
    k : int, optional
        no. of top movies to recommend, by default 10
    """

    with open(f'./data/{model}', 'rb') as f:
        nmf = pickle.load(f)

    with open('./data/movie_dict.bin', 'rb') as f:
        movie_id_dict = pickle.load(f)

    user_ratings = {key: int(value) for (key, value) in user_ratings.items() if value != ""}

    movie_titles = [i for i in movie_id_dict]
    movie_title_dict = {key: value for (value, key) in enumerate(movie_titles)}

    # 1. candiate generation

    # construct a user vector

    base_rating = [np.nan] * len(movie_titles)

    for key, value in user_ratings.items():
        index = movie_title_dict[key]
        base_rating[index] = value

    user_rating_list = np.array(base_rating).reshape(1,-1)

    # 2. scoring

    user_dataframe = pd.DataFrame(user_rating_list, index=['Recommendation'], columns=movie_titles).fillna(0)

    # calculate the score with the NMF model

    P_new_user = nmf.transform(user_dataframe)
    Q = nmf.components_

    R_new_user = np.dot(P_new_user, Q)

    user_dataframe = pd.DataFrame(R_new_user, index=['Recommendation'], columns=movie_titles)

    unrated_boolean = np.isnan(user_rating_list)[0]

    unrated_df = user_dataframe.iloc[:, unrated_boolean]

    # return the top-k highst rated movie ids or titles

    sorted_new_user_df = unrated_df.T.sort_values(by='Recommendation', ascending=False).head(k)

    movie_ids = [movie_id_dict[i] for i in sorted_new_user_df.index]
    movie_ids = pd.Series(movie_ids, index = sorted_new_user_df.index).to_frame(name="MovieId")

    final_recommendation = pd.concat([sorted_new_user_df, movie_ids], axis=1)

    return final_recommendation.index.to_list()[:k]


if __name__ == "__main__":
    user_rating = ask_user_recommendations()
    table = recommend_nmf(user_rating)
    print(table)