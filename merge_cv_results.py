from config import datasets, cv_types
import os

result_file = os.path.join("cv_results", "cv_results_agg.csv")
output_file = open(result_file, "w")
output_file.write("dataset, cv_setting, avg_cv_mean_accuracy, avg_cv_accuracy_stddev, avg_holdout_mean_accuracy, avg_holdout_accuracy_stddev, avg_time")
output_file.write("\n")

for dataset in datasets:
    for cv_type in cv_types:
        result_part_file = os.path.join("cv_results", dataset+"_"+str(cv_type)+".csv")
        if os.path.exists(result_part_file):
            with open(result_part_file) as f:
                output_file.write(f.readlines()[1])
                output_file.write("\n")
        else:
            print(f"{dataset} not completed")
