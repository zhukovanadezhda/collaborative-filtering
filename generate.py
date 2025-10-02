import numpy as np
import os
from tqdm import tqdm, trange
import argparse

from scripts.tools import complete_ratings, compute_rmse


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
    params = {
        "n_factors": 99,
        "n_iters": 41,
        "reg": 0.795,
        "random_state": 42
    }

    # Complete the ratings table
    table = complete_ratings(
        train_path="data/ratings_train.npy",
        test_path="data/ratings_test.npy",
        params=params
        )

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE