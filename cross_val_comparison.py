import time
import subprocess
from config import datasets, cv_types

for dataset in datasets:
    for cv_type in cv_types:
        subprocess.run(["jbsub",
            "-cores", "4",
            "-q", "x86_24h",
            "-mem", "128g",
            "-proj", "rasl_cross_val_comparison",
            "-name", f"rasl.cv.{dataset}-{cv_type}",
            "python", "run_cross_validate.py", f"{dataset}", f"{cv_type}"],
            capture_output=False)
        time.sleep(1)


