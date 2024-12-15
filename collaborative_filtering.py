import pandas as pd
import numpy as np
from time import time

eps = 1e-9

def find_co_rated_movie_indices(
        active_user_idx:int, 
        other_user_idx:int, 
        item_user_matrix:np.ndarray
    ):
    non_zero_columns = np.all(item_user_matrix[[active_user_idx, other_user_idx], :] != 0, axis=0)  # columns that are co-rated by the two users
    indices = np.where(non_zero_columns)[0].tolist()  # indices converted to list

    return indices


def user_based_correlation(
        active_user_idx:int, 
        other_user_idx:int, 
        co_rated_indices:list, 
        item_user_matrix:np.ndarray, 
        avg_rating_per_user:np.ndarray
    ):
    avg_active_rating = avg_rating_per_user[active_user_idx]
    avg_other_rating = avg_rating_per_user[other_user_idx]

    co_rated_movies = item_user_matrix[[active_user_idx, other_user_idx], :][:, co_rated_indices]

    numerator = np.dot((co_rated_movies[0, :] - avg_active_rating), (co_rated_movies[1, :] - avg_other_rating))
    denominator = np.dot(np.sqrt(((co_rated_movies[0, :] - avg_active_rating)**2).sum()), np.sqrt(((co_rated_movies[1, :] - avg_other_rating)**2).sum()))
    
    return numerator / (denominator + eps)

if __name__ == "__main__":
    df = pd.read_csv("ratings_small.csv")
    df.userId = df.userId.apply(lambda x: x - 1)  # subtract 1 from user ID

    idx2MovieId = {i: idx for i, idx in enumerate(df.movieId.value_counts().sort_index().index)}
    MovieId2idx = {idx: i for i, idx in enumerate(df.movieId.value_counts().sort_index().index)}

    num_movies = len(df.movieId.value_counts())
    num_users = len(df.userId.value_counts())

    print(f"Number of Movies = {num_movies}")
    print(f"Number of Users = {num_users}")

    item_user_matrix = np.zeros((num_users, num_movies))  # store item user rating matrix
    # fill values in a vectorized way
    user_indices = df.userId.astype(int).values
    movie_indices = df.movieId.map(MovieId2idx).values
    ratings = df.rating.values
    item_user_matrix[user_indices, movie_indices] = ratings

    num_movies_rated_per_user = (item_user_matrix > 0).sum(axis=1)

    avg_number_of_movies_rated = num_movies_rated_per_user.mean()
    max_number_of_movies_rated = num_movies_rated_per_user.max()
    min_number_of_movies_rated = num_movies_rated_per_user.min()

    print(f"Average Number of Movies Rated By Users = {avg_number_of_movies_rated}")
    print(f"Maximum Number of Movies Rated By Users = {max_number_of_movies_rated}")
    print(f"Minimum Number of Movies Rated By Users = {min_number_of_movies_rated}")

    avg_rating_per_user = item_user_matrix.sum(axis=1) / num_movies_rated_per_user  # mean(r_i)

    active_user_idx = int(input("Enter Active User Idx: "))

    movie_index_to_predict = int(input("Enter movie index to Predict: "))

    kappa = 0
    result_sum = 0

    start_time = time()
    for other_user_idx in range(0, num_users):
        if other_user_idx == active_user_idx:  # exclude active user idx
            continue
    
        # get co-rated indices
        co_rated_indices = find_co_rated_movie_indices(
            active_user_idx=active_user_idx, 
            other_user_idx=other_user_idx,  # ssample 3 has 4 co-rated movies
            item_user_matrix=item_user_matrix
        )

        if len(co_rated_indices) == 0:  # no co-rated movies
            continue

        w = user_based_correlation(
            active_user_idx=active_user_idx,
            other_user_idx=other_user_idx,
            co_rated_indices=co_rated_indices,
            item_user_matrix=item_user_matrix,
            avg_rating_per_user=avg_rating_per_user
        )

        # add absolute value of w to kappa
        if w >= 0:
            kappa += w
        else:
            kappa -= w

        if item_user_matrix[other_user_idx][movie_index_to_predict] != 0:
            result_sum += w * (item_user_matrix[other_user_idx][movie_index_to_predict] - avg_rating_per_user[other_user_idx])

    kappa = 1 / (kappa + eps)

    predicted_rating = avg_rating_per_user[active_user_idx] + kappa * result_sum

    end_time = time()

    print(f"\nElapsed time = {end_time - start_time}")
    print(f"User {active_user_idx} Predicted Rating on Movie {idx2MovieId[movie_index_to_predict]} = {predicted_rating}")       
