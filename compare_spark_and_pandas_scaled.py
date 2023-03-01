import argparse
from config import n_folds, n_batches_per_fold, get_large_data, get_pipeline_spark, get_pipeline, get_pipeline_sklearn, get_pipeline_sklearn_rasl, regression_datasets, classification_datasets, get_scorer, get_data
import time
import numpy as np
import os
import math
from lale.lib.rasl import Batching, PrioResourceAware, fit_with_batches
from sklearn.metrics import f1_score
from memory_profiler import profile
import resource
import pandas as pd
from config import datasets
from lale.datasets import pandas2spark
def get_metric(trained_pipeline, test_X, test_y):
    from sklearn.metrics import accuracy_score
    predictions = trained_pipeline.predict(test_X)
    result = accuracy_score(test_y.sort_index(), predictions)
    return result

def get_results(dataset, expt_type, random_seed, fraction):
    if expt_type=="spark":
        print("spark training to be started")
        import pdb;pdb.set_trace()
        (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="pandas", random_seed=random_seed)
        df = train_X
        df['label_column']=train_y
        df_subset = df.sample(frac=fraction, axis=0, replace=True, random_state=random_seed)#np.random.RandomState(np.random.randint(0, 100)))
        train_y=df_subset['label_column']
        train_X = df_subset.drop(columns='label_column')
        train_X = train_X.reset_index(drop=True)
        train_X = pandas2spark(train_X)
        pipeline = get_pipeline(dataset, convert_to_pandas = True, sort_index=True) 
        t1=time.time()
        import pdb;pdb.set_trace()
        try:
            trained_pipeline = pipeline.fit(train_X, train_y)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            import pdb;pdb.set_trace()
            print(f"Exception while running for {dataset}", e)
        t2 = time.time()
        import pdb;pdb.set_trace()
        print("Training complete")
        (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="spark", random_seed=random_seed)
        result = get_metric(trained_pipeline, test_X, test_y) 
    elif expt_type=="pandas":
        print("rasl fit to be started")
        (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="pandas", random_seed=random_seed)
        df = train_X
        df['label_column']=train_y
        df_subset = df.sample(frac=fraction, axis=0, replace=True, random_state=random_seed)
        train_y=df_subset['label_column']
        train_X = df_subset.drop(columns='label_column') 
        train_X = train_X.reset_index(drop=True)
        pipeline = get_pipeline(dataset, sort_index=True)
        t1=time.time()
        trained_pipeline = pipeline.fit(train_X, train_y)
        t2 = time.time()
        print("Training complete")
        result = get_metric(trained_pipeline, test_X, test_y) 
        #result = scoring_metric(trained_pipeline, test_X, test_y)
    output = {}
    output['test_score'] = result
    output['fit_time'] = t2-t1
    return output

if __name__=="__main__":
    header_row = "dataset, setting, fraction, holdout_accuracy_mean, holdout_accuracy_stddev, time_mean, time_stddev"
    with open(os.path.join("batching_results","spark_pandas_scaled_comparison_debug.csv"), "w") as f:
        f.write(header_row)
        f.write("\n")
    for dataset in datasets:
        random_seeds = [0,42,90,33,56]
        random_seed=42
        fractions =[8192]#1024, 2048, 4096, 8192]#[1/16, 1/8,1/4,1/2,1, 2, 4]#, 8]
        for fraction in fractions:
            accuracies_1 = []
            run_times_1 = []
            accuracies_2 = []
            run_times_2 = []
            try:
                scores_spark = get_results(dataset, "spark", random_seed=random_seed, fraction=fraction)
            except:
                scores_spark = {}
                scores_spark['test_score']=-1
                scores_spark['fit_time']=-1
            try:
                scores_pandas = get_results(dataset, "pandas", random_seed=random_seed, fraction=fraction)
                #assert scores_spark["test_score"] == scores_pandas["test_score"], "not equal"
            except BaseException as e:
                import traceback
                traceback.print_exc()
                print(f"Exception while running for {dataset}", e)
                scores_pandas = {}
                scores_pandas['test_score']=-1
                scores_pandas['fit_time'] = -1
            accuracies_1.append(scores_spark['test_score'])
            run_times_1.append(scores_spark['fit_time'])
            accuracies_2.append(scores_pandas['test_score'])
            run_times_2.append(scores_pandas['fit_time'])
            result_row_1 = dataset+", spark, "+str(fraction)+","+str(np.mean(accuracies_1))+","+str(np.std(accuracies_1))+","+str(np.mean(run_times_1))+","+str(np.std(run_times_1))+"\n"
            result_row_2 = dataset+", pandas, "+str(fraction)+","+str(np.mean(accuracies_2))+","+str(np.std(accuracies_2))+","+str(np.mean(run_times_2))+","+str(np.std(run_times_2))+"\n"
            with open(os.path.join("batching_results","spark_pandas_scaled_comparison_debug.csv"), "a") as f:
                f.write(result_row_1)
                f.write(result_row_2)        
