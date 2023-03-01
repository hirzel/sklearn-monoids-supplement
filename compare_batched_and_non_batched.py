import argparse
from config import n_folds, n_batches_per_fold, get_large_data, get_pipeline_spark, get_pipeline, get_pipeline_sklearn, get_pipeline_sklearn_rasl, regression_datasets, classification_datasets, get_scorer, get_data
import time
import numpy as np
import os
import math
from lale.lib.rasl import Batching, PrioResourceAware, fit_with_batches
from lale.lib.rasl import mockup_data_loader
from sklearn.metrics import f1_score
from memory_profiler import profile
import resource
import pandas as pd
from config import datasets

def get_metric(trained_pipeline, test_X, test_y):
    from sklearn.metrics import accuracy_score
    predictions = trained_pipeline.predict(test_X)
    result = accuracy_score(test_y, predictions)
    return result

def get_results(dataset, expt_type, random_seed):
    if expt_type=="batched":
        N_BATCHES=5
        print("batched training to be started")
        (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="pandas", random_seed=random_seed)
        pipeline = get_pipeline(dataset) 
        t1=time.time()
        from lale.lib.rasl import fit_with_batches, PrioResourceAware
        trained_pipeline = fit_with_batches(pipeline, mockup_data_loader(train_X, train_y, N_BATCHES, astype="pandas"), None, None, np.unique(test_y),
                    None, PrioResourceAware(), partial_transform=False, verbose=1, progress_callback=None)
        t2 = time.time()
        print("Training complete")
        result = get_metric(trained_pipeline, test_X, test_y) 
    elif expt_type=="non_batched":
        print("rasl fit to be started")
        (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="pandas", random_seed=random_seed)
        pipeline = get_pipeline(dataset)
        t1=time.time()
        trained_pipeline = pipeline.fit(train_X, train_y)
        t2 = time.time()
        print("Training complete")
        result = get_metric(trained_pipeline, test_X, test_y) 
    output = {}
    output['test_score'] = result
    output['fit_time'] = t2-t1
    return output

if __name__=="__main__":
    header_row = "dataset, setting, holdout_accuracy_mean, holdout_accuracy_stddev, time_mean, time_stddev"
    with open(os.path.join("batching_results","batched_non_batched_comparison.csv"), "w") as f:
        f.write(header_row)
        f.write("\n")
    for dataset in datasets:
        accuracies_1 = []
        run_times_1 = []
        accuracies_2 = []
        run_times_2 = []
        random_seeds=[0,42,90,33,56]
        for random_seed in random_seeds:
            try:
                scores_batched = get_results(dataset, "batched", random_seed=random_seed)
                scores_non_batched = get_results(dataset, "non_batched", random_seed=random_seed)
                print(scores_batched["test_score"], scores_non_batched["test_score"])
                assert scores_batched["test_score"] == scores_non_batched["test_score"], "not equal"
            except BaseException as e:
                import traceback
                traceback.print_exc()
                print(f"Exception while running for {dataset}", e)
                import pdb;pdb.set_trace()
            accuracies_1.append(scores_batched['test_score'])
            run_times_1.append(scores_batched['fit_time'])
            accuracies_2.append(scores_non_batched['test_score'])
            run_times_2.append(scores_non_batched['fit_time'])
        result_row_1 = dataset+", batched, "+str(np.mean(accuracies_1))+","+str(np.std(accuracies_1))+","+str(np.mean(run_times_1))+","+str(np.std(run_times_1))+"\n"
        result_row_2 = dataset+", non-batched, "+str(np.mean(accuracies_2))+","+str(np.std(accuracies_2))+","+str(np.mean(run_times_2))+","+str(np.std(run_times_2))+"\n"
        with open(os.path.join("batching_results","batched_non_batched_comparison.csv"), "a") as f:
            f.write(result_row_1)
            f.write(result_row_2)
