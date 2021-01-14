from hyperopt.hyperopt_complete import tune_complete


def optimize_all(num_trials):
    for source in range(1, 5):
        for target in range(1, 5):
            if not source == target:
                tune_complete(source, target, percent_broken=0.8, num_trials=num_trials)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="optimize hyperparams of all tasks")
    parser.add_argument(
        "--num_trials", required=True, type=int, help="number of trials per task"
    )
    opt = parser.parse_args()

    optimize_all(opt.num_trials)
