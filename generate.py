import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from scripts.als_tools import complete_ratings, compute_rmse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate a completed ratings table.'
        )
    parser.add_argument(
        "--name",
        type=str,
        default="ratings_eval.npy",
        help="Name of the npy of the ratings table to complete"
        )

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    
    # Define the parameters
    
    # Best parameters for the model without genres (found with Optuna)
    # params = {
    #     "n_factors": 99,
    #     "n_iters": 41,
    #     "reg": 0.795,
    #     "random_state": 42
    # }
    
    # Best parameters for the model with genres (found with Optuna)
    params = {
        "n_factors": 180,
        "n_iters": 52,
        "reg": 8.0,
        "random_state": 42
    }

    # Complete the ratings table
    table = complete_ratings(
        train_path="data/ratings_train.npy",
        test_path="data/ratings_test.npy",
        genres_path="data/genres.npy",
        params=params,
        merge=True
        )
    
    # # Print the RMSE on the test set
    # R_test = np.load("data/ratings_test.npy")
    # print(f"RMSE on the test set: {compute_rmse(R_test, table):.4f}")

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE