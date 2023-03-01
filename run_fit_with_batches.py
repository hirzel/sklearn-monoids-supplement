import argparse
from config import n_folds, n_batches_per_fold, get_large_data, get_pipeline, get_pipeline_sklearn, get_pipeline_sklearn_rasl, regression_datasets, classification_datasets, get_scorer
import time
import numpy as np
import os
import math
from lale.lib.lale import Batching
from sklearn.metrics import f1_score
from memory_profiler import profile
import resource
  
def limit_memory(maxsize):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

@profile
def run_fit_with_batches(split):
    scoring_metric = get_scorer(args.dataset, args.expt_type)
    #unique_class_labels = list(train_y.unique())
    pipeline = get_pipeline(args.dataset, "monoidal")
    #pipeline = get_pipeline(args.dataset, "incremental")
    limit_memory(5.4e9)
    if args.expt_type == 0:
        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, split=split)
        pipeline = get_pipeline_sklearn(args.dataset)
        print(pipeline)
        t1 = time.time()
        pipeline.fit(train_X, train_y)
        t2 = time.time()
        result = scoring_metric(pipeline, test_X, test_y)
    elif args.expt_type == 1:
        print("training to be started for split", split)
        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, astype="loader", batch_size=args.batch_size, split=split)
        batching = Batching(operator=pipeline, batch_size=args.batch_size, max_resident=200000000, inmemory=True)
        t1 = time.time()
        #trained_batching = batching.fit(train_X, train_y, classes=np.unique(test_y))
        trained_batching = batching.fit(train_X, train_y)
        t2 = time.time()
        print("Training complete")
        result = scoring_metric(trained_batching, test_X, test_y)
#        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, astype="loader", batch_size=args.batch_size, split=split)
#        pred = batching.predict(test_X)
#        f1=f1_score(test_y, pred, average='micro')
#        print(f"f1:{f1}", f1)
    elif args.expt_type==2:
        print("spark training to be started")
        (train_X, train_y), (test_X, test_y) = get_large_data(args.dataset, astype="spark", batch_size=args.batch_size, split=split)
        from lale.lib.rasl import PrioResourceAware, fit_with_batches
        #fit_with_batches(
        #    pipeline=pipeline,
        #    batches=train_X,  # type:ignore
        #    unique_class_labels=classes,
        #    max_resident=self.max_resident,
        #    prio=PrioResourceAware(),
        #    partial_transform=False,
        #    scoring=self.scoring,
        #    progress_callback=self.progress_callback,
        #    verbose=self.verbose,
        trained_pipeline = pipeline.fit(train_X, train_y)
        
    output = {}
    output['test_score'] = result
    output['fit_time'] = t2-t1
    return output

if __name__=="__main__":
    header_row = "dataset, setting, holdout_accuracy_mean, holdout_accuracy_stddev, time_mean, time_stddev"
    #parse command line args such as dataset name and cv type
    parser = argparse.ArgumentParser(
        description="Run batching fit_predict experiments"
    )
    parser.add_argument("dataset", metavar="D", type=str, help="Dataset to use for the experiment.")
    parser.add_argument("expt_type", metavar="EXPT_TYPE", type=int, help="experiment setting to use for the experiment. 1. sklearn, 2. rasl batching.")
    parser.add_argument("batch_size", metavar="BATCH_SIZE", type=int, help="batch size.")
    args = parser.parse_args()
    accuracies = []
    run_times = []
    for i in range(5):
        scores = run_fit_with_batches(i)
        accuracies.append(scores['test_score'])
        run_times.append(scores['fit_time'])
    result_row = args.dataset+","+str(args.expt_type)+","+str(np.mean(accuracies))+","+str(np.std(accuracies))+","+str(np.mean(run_times))+","+str(np.std(run_times))
    with open(os.path.join("batching_results",f"{args.dataset}_{args.expt_type}.csv"), "w") as f:
        f.write(header_row)
        f.write("\n")
        f.write(result_row)
