import argparse
from config import n_folds, n_batches_per_fold, get_pipeline, get_pipeline_sklearn, get_pipeline_sklearn_rasl, regression_datasets, classification_datasets, get_scorer
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
        from sklearn.model_selection import cross_validate, KFold
        result = cross_validate(pipeline, train_X, train_y, scoring=scoring_metric, cv=KFold(n_folds), return_estimator=True, n_jobs=1, pre_dispatch=1)
    elif args.cv_type == 1:
        from sklearn.model_selection import KFold
        from lale.lib.rasl import cross_validate, PrioBatch
        from lale.helpers import create_data_loader
        batches = create_data_loader(train_X, train_y, len(train_y))
        result= cross_validate(pipeline, batches,
                         scoring_metric, KFold(n_folds), unique_class_labels, None, prio=PrioBatch(), same_fold=True, verbose=1, return_estimator=True)
    elif args.cv_type == 2:
        from sklearn.model_selection import KFold
        from lale.lib.rasl import cross_validate, PrioBatch
        from sklearn.metrics import make_scorer, accuracy_score
        accuracy_scorer = make_scorer(accuracy_score)
        from lale.helpers import create_data_loader
        batches = create_data_loader(train_X, train_y, math.ceil(len(train_y)/n_batches_per_fold))
        result= cross_validate(pipeline, batches,
                         scoring_metric, KFold(n_folds), unique_class_labels, None, prio=PrioBatch(), same_fold=True, verbose=1, return_estimator=True)
    elif args.cv_type == 3:
        from sklearn.model_selection import KFold
        from lale.lib.rasl import cross_validate, PrioBatch
        from sklearn.metrics import make_scorer, accuracy_score
        accuracy_scorer = make_scorer(accuracy_score)
        from lale.helpers import create_data_loader
        batches = create_data_loader(train_X, train_y, math.ceil(len(train_y)/n_batches_per_fold))
        result = cross_validate(pipeline, batches,
                         scoring_metric, KFold(n_folds), unique_class_labels, None, prio=PrioBatch(), same_fold=False, verbose=1, return_estimator=True)
    trained_estimators = result['estimator']
    output = {}
    output['test_scores'] = result['test_score']
    holdout_scores = []
    for estimator in trained_estimators:
        holdout_scores.append(scoring_metric(estimator, test_X, test_y))
    output['holdout_scores']=holdout_scores
    return output

if __name__=="__main__":
    header_row = "dataset, cv_setting, avg_cv_mean_accuracy, avg_cv_accuracy_stddev, avg_holdout_mean_accuracy, avg_holdout_accuracy_stddev, avg_time"
    #parse command line args such as dataset name and cv type
    parser = argparse.ArgumentParser(
        description="Run CV experiments"
    )
    parser.add_argument("dataset", metavar="D", type=str, help="Dataset to use for the experiment.")
    parser.add_argument("cv_type", metavar="CV_TYPE", type=int, help="cross_val_score variation to use for the experiment. See config.py for what each value means.")
    args = parser.parse_args()
    cv_mean_accuracies = []
    cv_accuracy_stddevs = []
    holdout_mean_accuracies = []
    holdout_accuracy_stddevs = []
    cv_times = []
    for i in range(5):#repeat runs
        t1 = time.time()
        scores = run_cross_val_score()
        t2= time.time()
        cv_mean_accuracies.append(np.mean(scores['test_scores']))
        cv_accuracy_stddevs.append(np.std(scores['test_scores']))
        holdout_mean_accuracies.append(np.mean(scores['holdout_scores']))
        holdout_accuracy_stddevs.append(np.std(scores['holdout_scores']))
        cv_times.append(t2-t1)
    result_row = args.dataset+","+str(args.cv_type)+","+str(np.mean(cv_mean_accuracies))+","+str(np.mean(cv_accuracy_stddevs))+","+str(np.mean(holdout_mean_accuracies))+","+str(np.mean(holdout_accuracy_stddevs))+","+str(np.mean(cv_times))
    with open(os.path.join("cv_results",f"{args.dataset}_{args.cv_type}.csv"), "w") as f:
        f.write(header_row)
        f.write("\n")
        f.write(result_row)
