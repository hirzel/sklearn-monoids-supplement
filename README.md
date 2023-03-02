# sklearn-monoids-supplement

Supplemental material for paper submission
"Combining Machine Learning Pipelines with Monoids".

The implementation of the algorithms described in the paper is
available as part of the [Lale](https://github.com/ibm/lale)
open-source project.
Specifically, the relevant code can be found in the
[lale.lib.rasl](https://github.com/IBM/lale/tree/master/lale/lib/rasl)
folder of the Lale repository.

## Installation

Install [Lale](https://github.com/IBM/lale/blob/master/docs/installation.rst) and packages required for the evaluation:
```
pip install "lale[full,test]" memory_profiler
```

Download the following datasets in the `./rasl_datasets` directory:
- [KDD Cup 1999](https://www.openml.org/search?type=data&status=active&id=42746)
- [Steam Review](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)
- [eCommerce october 2019](https://www.kaggle.com/code/danofer/ecommerce-store-predict-purchases-data-prep/data?select=2019-Oct.csv)
- [Chicago data portal: Taxi Trips](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew)


## Evaluation

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
