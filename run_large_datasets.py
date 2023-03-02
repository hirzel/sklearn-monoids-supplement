import argparse
from config import n_folds, n_batches_per_fold, get_large_data, get_pipeline_spark, get_pipeline, get_pipeline_sklearn_large, get_pipeline_sklearn_rasl, regression_datasets, classification_datasets, get_scorer, get_data, get_pipeline_large
import time
import numpy as np
import os
import math
from lale.lib.rasl import Batching, PrioResourceAware, fit_with_batches
from sklearn.metrics import f1_score
from memory_profiler import profile
import resource
import pandas as pd

def limit_memory(maxsize):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

def run_fit_with_batches(split):
    scoring_metric = get_scorer(args.dataset, args.expt_type)
    #unique_class_labels = list(train_y.unique())
    #pipeline = get_pipeline(args.dataset, "monoidal")
    #pipeline = get_pipeline(args.dataset, "incremental")
    limit_memory(args.process_memory_limit)
    if args.expt_type == 0:
        print("Running the sklearn part")
        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, split=split)
        pipeline = get_pipeline_sklearn_large(args.dataset)
        print(pipeline)
        t1 = time.time()
        pipeline.fit(train_X, train_y)
        t2 = time.time()
        result = scoring_metric(pipeline, test_X, test_y)
    elif args.expt_type == 1:
        print("training to be started for split", split)
        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, astype="loader", batch_size=args.batch_size, split=split)
        pipeline = get_pipeline_large(args.dataset, "incremental")
        print("3HERE!!!")
        t1 = time.time()
        #trained_batching = batching.fit(train_X, train_y, classes=np.unique(test_y))
        #trained_batching = batching.fit(train_X, train_y)
        from lale.lib.rasl import fit_with_batches, PrioResourceAware
        trained = fit_with_batches(pipeline,batches_train= train_X, batches_valid=None, scoring=None, unique_class_labels=None, #np.unique(train_y),
                                           max_resident=args.max_resident, prio=PrioResourceAware(), partial_transform=False, verbose=0, progress_callback=None)
        t2 = time.time()
        print("Training complete")
        #result = scoring_metric(trained_batching, test_X, test_y)
        test_X_df = None
        for x, _ in test_X:
            test_X_df = pd.concat([test_X_df, x])
        result = scoring_metric(trained, test_X_df, test_y)
#        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, astype="loader", batch_size=args.batch_size, split=split)
#        pred = batching.predict(test_X)
#        f1=f1_score(test_y, pred, average='micro')
#        print(f"f1:{f1}", f1)
    output = {}
    output['test_score'] = result
    output['fit_time'] = t2-t1
    return output

if __name__=="__main__":
    header_row = "dataset, setting, batch_size, process_memory_limit, max_resident, holdout_accuracy_mean, holdout_accuracy_stddev, time_mean, time_stddev"
    #parse command line args such as dataset name and cv type
    parser = argparse.ArgumentParser(
        description="Run batching fit_predict experiments"
    )
    parser.add_argument("dataset", metavar="D", type=str, help="Dataset to use for the experiment.")
    parser.add_argument("expt_type", metavar="EXPT_TYPE", type=int, help="experiment setting to use for the experiment. 1. sklearn, 2. rasl batching.")
    parser.add_argument("batch_size", metavar="BATCH_SIZE", type=int, help="batch size.")
    parser.add_argument("process_memory_limit", metavar="process_memory_limit", type=int, help="")
    parser.add_argument("max_resident", metavar="max_resident", type=int, help="")

    args = parser.parse_args()
    accuracies = []
    run_times = []
    for i in range(1):
        try:
            scores = run_fit_with_batches(i)
        except BaseException as e:
            import traceback
            traceback.print_exc()
            print("Exception while running run_fit_with_batches", e)
            scores={}
            scores['test_score']=-1
            scores['fit_time']=-1
        accuracies.append(scores['test_score'])
        run_times.append(scores['fit_time'])
    result_row = args.dataset+","+str(args.expt_type)+","+str(args.batch_size)+","+"{:e}".format(args.process_memory_limit)+","+"{:e}".format(args.max_resident)+","+str(np.mean(accuracies))+","+str(np.std(accuracies))+","+str(np.mean(run_times))+","+str(np.std(run_times))
    with open(os.path.join("batching_results",f"{args.dataset}_{args.expt_type}_{args.batch_size}_"+"{:e}".format(args.process_memory_limit)+"_"+"{:e}".format(args.max_resident)+".csv"), "w") as f:
        f.write(header_row)
        f.write("\n")
        f.write(result_row)
