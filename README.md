# sklearn-monoids-supplement

Supplemental material for paper submission
"Combining Machine Learning Pipelines with Monoids".

The implementation of the algorithms described in the paper is
available as part of the [Lale](https://github.com/ibm/lale)
open-source project.
Specifically, the relevant code can be found in the
[lale.lib.rasl](https://github.com/IBM/lale/tree/master/lale/lib/rasl)
folder of the Lale repository.
In addition to the Lale repository, this supplemental repository provides
the scripts for running the experiments described in the paper.
Running these requires installing Lale, as well as downloading the
datasets, all of which are publicly available and referenced from the
paper.

#### RQ1. Can batching enable fitting pipelines on larger data without Spark SQL?

See [run_large_datasets.py](run_large_datasets.py).

#### RQ2. When should you use the pandas backend and when the Spark SQL backend?

See [compare_spark_and_pandas_scaled.py](compare_spark_and_pandas_scaled.py).

#### RQ3. Do the pandas and Spark SQL backends yield identical results to sklearn?

The relevant tests are in the Lale repository under
[test_relational_sklearn.py](https://github.com/IBM/lale/blob/master/test/test_relational_sklearn.py)
and
[test_relational_from_sklearn_manual.py](https://github.com/IBM/lale/blob/master/test/test_relational_from_sklearn_manual.py).

#### RQ4. Does batched execution yield identical results as non-batched?

See [compare_batched_and_non_batched.py](compare_batched_and_non_batched.py).

#### RQ5. How much accuracy does partial-transform training lose?

See [partialtfm.ipynb](partialtfm.ipynb).

#### RQ6. How effective is out-of-fold cross-validation at picking models?

See [crossval.ipynb](crossval.ipynb).
