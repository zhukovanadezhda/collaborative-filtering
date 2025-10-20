import numpy as np
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
    table = np.load(args.name)
    print('Ratings Loaded.')
    
    # Define the parameters
    
    # Best parameters for the model without genres (found with Optuna)
    # params = {
    #     "n_factors": 99,
    #     "n_iters": 41,
    #     "reg": 0.795,
    #     "random_state": 42
    # }
    
    # # Best parameters for the model with genres (found with Optuna)
    # params = {
    #     "n_factors": 180, 
    #     "n_iters": 52,
    #     "reg": 8.0,
    #     "random_state": 42
    # }

    # Best parameters for the laplacian graph regularization
    # params = {
    #     "n_factors": 50,
    #     "n_iters": 20,
    #     "reg": 0.8,
    #     "random_state": 42,
    #     "S_topk": 10,
    #     "S_eps": 1e-5,
    #     "alpha": 0.7
    # }

    # # Parameters for the new model (maybe not optimal)
    # params = {
    #     'n_factors': 1,
    #     'n_iters': 21,
    #     'lambda_u': 7,
    #     'lambda_v': 13,
    #     'lambda_wg': 510,
    #     'lambda_wy': 1e+4,
    #     'lambda_bu': 2.5,
    #     'lambda_bi': 6.5,
    #     'year_mode': 'bins',
    #     'n_year_bins': 1,
    #     'update_w_every': 21
    # }

    # Parameters for new laplacian graph regularization
    params = {
        'n_factors': 150,
        'n_iters': 35,
        'lambda_u': 5,
        'lambda_v': 4,
        'lambda_wg': 5,
        'lambda_wy': 0,
        'lambda_bu': 3,
        'lambda_bi': 2,
        'S_topk': 49,
        'S_eps': 3e-05,
        'alpha': 0.3
        }

    # Complete the ratings table
    table = complete_ratings(
        train_path="data/ratings_train.npy",
        test_path="data/ratings_test.npy",
        genres_path="data/genres.npy",
        years_path="data/years.npy",
        params=params,
        merge=False,
        use_laplacian=True,
    )

    # Print the RMSE on the test set
    R_test = np.load("data/ratings_test.npy")
    print(f"RMSE on the test set: {compute_rmse(R_test, table):.4f}")

    # Save the completed table 
    np.save("output.npy", table)