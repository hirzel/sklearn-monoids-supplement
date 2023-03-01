import argparse
from config import n_folds, n_batches_per_fold, get_pipeline, get_pipeline_sklearn, regression_datasets, classification_datasets, get_scorer
import time
import numpy as np
import os
import math

def run_cross_val_score():
    import lale.datasets.openml
    scoring_metric = get_scorer(args.dataset, args.cv_type)
    if args.dataset in regression_datasets:
        (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(args.dataset, "regression", preprocess=False)
    else:
        (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(args.dataset, "classification", preprocess=False)
    unique_class_labels = list(train_y.unique())
    pipeline = get_pipeline(args.dataset)
    if args.cv_type == 0:
        pipeline = get_pipeline_sklearn(args.dataset)
        from sklearn.model_selection import cross_val_score, KFold
        return cross_val_score(pipeline, train_X, train_y, scoring=scoring_metric, cv=KFold(n_folds))
    elif args.cv_type == 1:
        from lale.lib.rasl import cross_val_score, PrioBatch
        from lale.helpers import create_data_loader
        batches = create_data_loader(train_X, train_y, math.ceil(len(train_y)/n_folds))
        #from lale.lib.rasl import mockup_data_loader
        #batches=mockup_data_loader(train_X, train_y, n_folds)
        return cross_val_score(pipeline, batches, n_folds, n_folds, 1,
                         scoring_metric, unique_class_labels, PrioBatch(), same_fold=True, verbose=1)
    elif args.cv_type == 2:
        from lale.lib.rasl import cross_val_score, PrioBatch
        from sklearn.metrics import make_scorer, accuracy_score
        accuracy_scorer = make_scorer(accuracy_score)
        from lale.helpers import create_data_loader
        batches = create_data_loader(train_X, train_y, math.ceil(len(train_y)/(n_batches_per_fold*n_folds)))
        return cross_val_score(pipeline, batches, len(batches), n_folds, n_batches_per_fold,
                         scoring_metric, unique_class_labels, PrioBatch(), same_fold=True, verbose=1)
    elif args.cv_type == 3:
        from lale.lib.rasl import cross_val_score, PrioBatch
        from sklearn.metrics import make_scorer, accuracy_score
        accuracy_scorer = make_scorer(accuracy_score)
        from lale.helpers import create_data_loader
        batches = create_data_loader(train_X, train_y, math.ceil(len(train_y)/(n_batches_per_fold*n_folds)))
        return cross_val_score(pipeline, batches, len(batches), n_folds, n_batches_per_fold,
                         scoring_metric, unique_class_labels, PrioBatch(), same_fold=False, verbose=1)

if __name__=="__main__":
    header_row = "dataset, cv_setting, avg_cv_mean_accuracy, avg_cv_accuracy_stddev, avg_time"
    #parse command line args such as dataset name and cv type
    parser = argparse.ArgumentParser(
        description="Run CV experiments"
    )
    parser.add_argument("dataset", metavar="D", type=str, help="Dataset to use for the experiment.")
    parser.add_argument("cv_type", metavar="CV_TYPE", type=int, help="cross_val_score variation to use for the experiment. See config.py for what each value means.")
    args = parser.parse_args()
    cv_mean_accuracies = []
    cv_accuracy_stddevs = []
    cv_times = []
    for i in range(5):#repeat runs
        t1 = time.time()
        scores = run_cross_val_score()
        t2= time.time()
        cv_mean_accuracies.append(np.mean(scores))
        cv_accuracy_stddevs.append(np.std(scores))
        cv_times.append(t2-t1)
    result_row = args.dataset+","+str(args.cv_type)+","+str(np.mean(cv_mean_accuracies))+","+str(np.mean(cv_accuracy_stddevs))+","+str(np.mean(cv_times))
    with open(os.path.join("cv_results",f"{args.dataset}_{args.cv_type}.csv"), "w") as f:
        f.write(header_row)
        f.write("\n")
        f.write(result_row)
