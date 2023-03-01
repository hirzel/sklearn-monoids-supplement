import time
import subprocess
from config import large_datasets, pipeline_types

for dataset in large_datasets:
    for pipeline_type in pipeline_types:
        subprocess.run(["jbsub",
            "-cores", "4",
            "-q", "x86_24h",
            "-mem", "164g",
            "-proj", "rasl_fit_with_batches_comparison",
            "-name", f"rasl.cv.{dataset}-{pipeline_type}",
            "python", "run_fit_with_batches.py", f"{dataset}", f"{pipeline_type}", "25000"],
            capture_output=False)
        time.sleep(1)


