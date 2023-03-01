import numpy as np
import pandas as pd
import os
from lale.lib.lale import Tee
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from lale.datasets.data_schemas import SparkDataFrameWithIndex
from lale.lib.rasl import Alias
from lale.lib.rasl import Convert, SortIndex

RANDOM_SEED=32

large_datasets_dir="./rasl_datasets"
large_datasets=[
        "ecommerce_2019_oct",
        "ecommerce_2019_oct_clean",
        "steam_reviews_no_review_clean",
        #        "kddcup99full"
        "data_cityofchicago_org___wrvz-psew___0_0_"
        ]
pipeline_types=[0,#sklearn,
            1#rasl_monoidal
            ]
datasets_expts_1 = ["airlines_delay"]
datasets = [
#    "spectf",  # 267 x 44, all numeric
#    "breast-cancer",  # 286 x 9, all categorical
#    "Australian",  # 690
#    "blood-transfusion-service-center",  # 748
#    "diabetes",  # 768
#    "credit-g",  # 1,000 x 20, 13 categorical + 7 numeric
#    "car",  # 1,728
#    "mfeat-factors",  # 2,000
#    "kc1",  # 2,109
#    "kr-vs-kp",  # 3,196
    "sylvine",  # 5,124
    "phoneme",  # 5,404
#    "jungle_chess_2pcs_raw_endgame_complete",  # 44,819
#    "shuttle",  # 58,000
    #"kddcup99full",  # 4,898,431
    #"airlines_delay"],  # 10,000,000
]

regression_datasets = ["airlines_delay", "ecommerce_2019_oct", "ecommerce_2019_oct_clean", "data_cityofchicago_org___wrvz-psew___0_0"]
classification_datasets = ["credit-g",
        "Australian",
        "blood-transfusion-service-center",
        "breast-cancer",
        "car",
        "diabetes",
        "jungle_chess_2pcs_raw_endgame_complete",
        "kc1",
        "kr-vs-kp",
        "mfeat-factors",
        "phoneme",
        "shuttle",
        "spectf",
        "sylvine",
        "kddcup99full",
        "steam_reviews_no_review_clean"]

dataset_to_cat_columns = {"credit-g":['checking_status', 'credit_history', 'purpose', 
               'savings_status', 'employment', 'personal_status', 'other_parties', 
               'property_magnitude', 'other_payment_plans', 'housing', 'job',
               'own_telephone', 'foreign_worker'],
               "airlines_delay":["month", "dayofmonth", "dayofweek", "uniquecarrier", "origin", "dest"],
               "ecommerce_2019_oct":['event_type', 'category_id', 'category_code','user_id'],
               "ecommerce_2019_oct_clean":['event_time', 'event_type', 'product_id','category_id', 'category_code','brand'],
               'kddcup99full':['protocol_type', 'service', 'flag', 'land'],
               'steam_reviews_no_review_clean':['app_id', 'app_name', 'language', 'written_during_early_access','steam_purchase', 'received_for_free'],
               'data_cityofchicago_org___wrvz-psew___0_0':['Tolls', 'Payment Type','Company']}
dataset_to_num_columns = {"credit-g": ['duration', 'credit_amount', 'installment_commitment', 'residence_since','age', 
              'existing_credits', 'num_dependents'],
              "airlines_delay":["crsdeptime", "crsarrtime", "distance"],
              "ecommerce_2019_oct":['price'],
              "ecommerce_2019_oct_clean":[],
              'kddcup99full':['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 
                  'logged_in', 'lnum_compromised', 'lroot_shell', 'lsu_attempted', 'lnum_root', 'lnum_file_creations',
'lnum_shells', 'lnum_access_files', 'lnum_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count','serror_rate', 'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'],
              'steam_reviews_no_review_clean':['votes_helpful', 'votes_funny', 'weighted_vote_score', 'comment_count',
                  'author.num_games_owned', 'author.num_reviews', 'author.playtime_forever', 'author.playtime_last_two_weeks', 'author.playtime_at_review'],
              'data_cityofchicago_org___wrvz-psew___0_0':['Trip Seconds','Trip Miles','Pickup Community Area','Dropoff Community Area']}

dataset_to_max_values = {
    "airlines_delay":400,
    "default":20
}
def get_data(dataset, astype="pandas", batch_size=None, test_size=0.33, random_seed=RANDOM_SEED):
    import lale.datasets.openml
    from lale.lib.rasl.datasets import csv_data_loader
    if dataset in regression_datasets:
        task_type="regression"
    else:
        task_type="classification"
    if astype=="loader":
        assert batch_size is not None
        train_X = csv_data_loader(os.path.join("./datasets", dataset+"_train.csv"), label_name='label', rows_per_batch=batch_size)
        train_y = None
        test_X = csv_data_loader(os.path.join("./datasets", dataset+"_test.csv"), label_name='label', rows_per_batch=batch_size)
        test_y = None
        for _, y in test_X:
            if test_y is None:
                test_y = y
            else:
                test_y = pd.concat([test_y, y])
        test_X = csv_data_loader(os.path.join("./datasets", dataset+"_test.csv"), label_name='label', rows_per_batch=batch_size)
    #elif astype=="big_spark":

    else:
        (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(dataset, task_type, test_size=test_size, preprocess=False, astype=astype, seed=random_seed)
    return (train_X, train_y), (test_X, test_y)

def get_large_data(dataset, astype="pandas", batch_size=None, label_name='label', split=0):
    import lale.datasets.openml
    from lale.lib.rasl.datasets import csv_data_loader
    if dataset in regression_datasets:
        task_type="regression"
    else:
        task_type="classification"
    if astype=="loader":
        assert batch_size is not None
        train_X = csv_data_loader(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_train.csv"), label_name=label_name, rows_per_batch=batch_size)
        train_y = None
        test_X = csv_data_loader(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_test.csv"), label_name=label_name, rows_per_batch=batch_size)
        test_y = None
        for _, y in train_X:
            if train_y is None:
                train_y = y
            else:
                train_y = pd.concat([train_y, y])
        train_X = csv_data_loader(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_train.csv"), label_name=label_name, rows_per_batch=batch_size)
        for _, y in test_X:
            if test_y is None:
                test_y = y
            else:
                test_y = pd.concat([test_y, y])
        test_X = csv_data_loader(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_test.csv"), label_name='label', rows_per_batch=batch_size)
    elif astype=="spark":
        #conf=SparkConf()
        #conf.set("spark.executor.memory", "64g")
        #conf.set("spark.driver.memory", "64g")
        #spark = SparkContext.getOrCreate(conf)
        try:
            spark = SparkSession.builder.master("local[1]").appName("RASL").config("spark.driver.memory", "64g").getOrCreate()
            train_X = spark.read.option("header",True).csv(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_train.csv"))
            train_X=SparkDataFrameWithIndex(train_X)
            train_y = train_X.select(label_name, "index")
            train_X.drop(label_name)
            train_X=SparkDataFrameWithIndex(train_X, index_names=["index"])
            train_y=SparkDataFrameWithIndex(train_y, index_names=["index"])

            test_X = spark.read.option("header",True).csv(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_test.csv"))
            test_X=SparkDataFrameWithIndex(test_X)
            test_y = test_X.select(label_name, "index")
            test_X.drop(label_name)
            test_X=SparkDataFrameWithIndex(test_X, index_names=["index"])
            test_y=SparkDataFrameWithIndex(test_y, index_names=["index"])
        except:
            (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="spark-with-index")
    else:
        try:
            import pdb;pdb.set_trace()
            train_X = pd.read_csv(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_train.csv"))
            train_y = train_X[label_name]
            train_X.drop([label_name], axis=1)
            test_X = pd.read_csv(os.path.join(large_datasets_dir, "splits/split"+str(split), dataset+"_test.csv"))
            test_y = test_X[label_name]
            test_X.drop([label_name], axis=1)
        except:
            (train_X, train_y), (test_X, test_y) = get_data(dataset, astype="pandas")
    return (train_X, train_y), (test_X, test_y)

def get_estimator(dataset, estimator_type, module_type):
    if dataset in regression_datasets:
        est = get_regressor(estimator_type, module_type)
    else:
        est = get_classifier(estimator_type, module_type)
    return est

def get_regressor(estimator_type, module_type):
    if module_type=="lale":
        from lale.lib.sklearn import RandomForestRegressor
        from lale.lib.sklearn import SGDRegressor
    elif module_type=="sklearn":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import SGDRegressor
    if estimator_type is None:
        return RandomForestRegressor(random_state=42)
    elif estimator_type == "incremental":
        return SGDRegressor()
    elif estimator_type == "monoidal":
        from lale.lib.snapml import BatchedTreeEnsembleRegressor
        from lale.lib.lightgbm import LGBMRegressor
        return Convert(astype="numpy") >> BatchedTreeEnsembleRegressor(base_ensemble=LGBMRegressor())
    else:
        raise ValueError(f"estimator type {estimator_type} is not valid")

def get_classifier(estimator_type, module_type):
    if module_type=="lale":
        from lale.lib.sklearn import RandomForestClassifier
        from lale.lib.sklearn import SGDClassifier
        from lale.lib.snapml import BatchedTreeEnsembleClassifier
    elif module_type=="sklearn":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import SGDClassifier
    if estimator_type is None:
        return RandomForestClassifier(random_state=42)
    elif estimator_type == "incremental":
        #return BatchedTreeEnsembleClassifier()
        return SGDClassifier()
    elif estimator_type == "monoidal":
        from lale.lib.rasl import BatchedBaggingClassifier
        from lale.lib.sklearn import LogisticRegression
        from lale.lib.snapml import BatchedTreeEnsembleClassifier
        #return Convert(astype="numpy") >> BatchedTreeEnsembleClassifier()
        return BatchedBaggingClassifier()
    else:
        raise ValueError(f"estimator type {estimator_type} is not valid")

def get_cat_num_columns(train_X):
    from lale.lib.rasl.project import _columns_schema_to_list
    cat_columns = _columns_schema_to_list(train_X, {"type":"string"})
    num_columns = list(set(train_X.columns)-set(cat_columns))
    return cat_columns, num_columns

def dataset_to_cat_and_num_columns(dataset):
    try:
        (train_X, train_y), (test_X, test_y) = get_data(dataset)
        cat_columns, num_columns = get_cat_num_columns(train_X)
    except:
        cat_columns = dataset_to_cat_columns[dataset]
        num_columns = dataset_to_num_columns[dataset]
    return cat_columns, num_columns


def f(X, y):
    pass
#    if X.isnull().values.any():
#        print("XXXXXXXXXX X")
#    if y is not None and y.isnull().values.any():
#        print("XXXXXXXXXX y")

def get_prefix_rasl(dataset, clip):
    from lale.lib.rasl import SimpleImputer
    from lale.lib.rasl import HashingEncoder
    from lale.lib.rasl import MinMaxScaler
    from lale.lib.lale import ConcatFeatures
    from lale.lib.lale import Project
    cat_columns, num_columns = dataset_to_cat_and_num_columns(dataset)
    cat_prep = None
    num_prep = None
    if cat_columns is not None and len(cat_columns) != 0:
        cat_proj = Project(columns=cat_columns)
        cat_prep = SimpleImputer(strategy="constant") >> HashingEncoder()
    if num_columns is not None and len(num_columns) != 0:
        num_proj = Project(columns=num_columns)
        num_prep = SimpleImputer(strategy="mean") >> MinMaxScaler(clip=clip)
    if cat_prep is not None and num_prep is not None:
        prefix = (
            (cat_proj >> cat_prep) & (num_proj >> num_prep)
        ) >> ConcatFeatures
    elif cat_prep is not None:
        prefix = cat_prep
    else:
        prefix = num_prep
    assert prefix is not None
    return prefix


def get_prefix_sklearn(dataset, clip):
    from sklearn.impute import SimpleImputer
    from category_encoders.hashing import HashingEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    cat_columns, num_columns = dataset_to_cat_and_num_columns(dataset)
    prefix = ColumnTransformer([
        ("prep_cat",
         make_pipeline(SimpleImputer(strategy="constant"), HashingEncoder()),
         cat_columns),
        ("prep_num",
         make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler(clip=clip)),
         num_columns)
    ])
    return prefix

def get_pipeline(dataset, estimator_type=None, convert_to_pandas=False, sort_index=False):
    from lale.lib.rasl import SimpleImputer
    from lale.lib.rasl import OrdinalEncoder, HashingEncoder
    from lale.lib.rasl import MinMaxScaler
    from lale.lib.rasl import SelectKBest
    from lale.lib.sklearn import RandomForestClassifier, RandomForestRegressor
    from lale.lib.lale import ConcatFeatures
    from lale.lib.lale import Project, categorical
#    cat_columns = dataset_to_cat_columns[dataset]
#    num_columns = dataset_to_num_columns[dataset]

    cat_columns, num_columns = dataset_to_cat_and_num_columns(dataset)
    print(cat_columns, num_columns)
    k=10#np.min([10, train_X.shape[1]])
    est = get_estimator(dataset, estimator_type, "lale")
    cat_prep = None
    num_prep = None
    if convert_to_pandas:#using this flag to also conclude that this is a spark pipeline
        missing_values=None
    else:
        missing_values = np.nan
    if cat_columns is not None and len(cat_columns) !=0:
        cat_prep = (Project(columns=cat_columns) >> SimpleImputer(strategy="constant", missing_values=missing_values) 
            #>> OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            >> HashingEncoder())
    if num_columns is not None and len(num_columns) !=0:
        num_prep = (Project(columns=num_columns)
            >> SimpleImputer(strategy="mean")
            >> MinMaxScaler())
    if cat_prep is not None and num_prep is not None:
        pipeline = (cat_prep & num_prep) >> ConcatFeatures()
    elif cat_prep is None and num_prep is not None:
        pipeline = num_prep 
    elif cat_prep is not None and num_prep is None:
        pipeline = cat_prep
    if sort_index:
        pipeline = pipeline >> SortIndex()
    if convert_to_pandas:
        pipeline = pipeline >> Convert()
    pipeline = pipeline >> est
    return pipeline


def get_pipeline_large(dataset, estimator_type=None, convert_to_pandas=False, sort_index=False):
    from lale.lib.rasl import SimpleImputer
    from lale.lib.rasl import OrdinalEncoder, HashingEncoder
    from lale.lib.rasl import MinMaxScaler
    from lale.lib.rasl import SelectKBest
    from lale.lib.sklearn import RandomForestClassifier, RandomForestRegressor
    from lale.lib.lale import ConcatFeatures
    from lale.lib.lale import Project, categorical
#    cat_columns = dataset_to_cat_columns[dataset]
#    num_columns = dataset_to_num_columns[dataset]

    cat_columns, num_columns = dataset_to_cat_and_num_columns(dataset)
    print(cat_columns, num_columns)
    k=np.min([10, len(cat_columns)+len(num_columns)])
    est = get_estimator(dataset, estimator_type, "lale")
    cat_prep = None
    num_prep = None
    if convert_to_pandas:#using this flag to also conclude that this is a spark pipeline
        missing_values=None
    else:
        missing_values = np.nan
    if cat_columns is not None and len(cat_columns) !=0:
        cat_prep = (Project(columns=cat_columns) >> SimpleImputer(strategy="constant", missing_values=missing_values) 
            #>> OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            >> HashingEncoder())
    if num_columns is not None and len(num_columns) !=0:
        num_prep = (Project(columns=num_columns)
            >> SimpleImputer(strategy="mean")
            >> MinMaxScaler())
    if cat_prep is not None and num_prep is not None:
        pipeline = (cat_prep & num_prep) >> ConcatFeatures()
    elif cat_prep is None and num_prep is not None:
        pipeline = num_prep 
    elif cat_prep is not None and num_prep is None:
        pipeline = cat_prep
    if sort_index:
        pipeline = pipeline >> SortIndex()
    if convert_to_pandas:
        pipeline = pipeline >> Convert()
    pipeline = pipeline >> SelectKBest(k=k) >> est
    return pipeline

def get_pipeline_spark(dataset, estimator_type=None):
    from lale.lib.rasl import SimpleImputer
    from lale.lib.rasl import OrdinalEncoder, HashingEncoder
    from lale.lib.rasl import MinMaxScaler
    from lale.lib.rasl import SelectKBest
    from lale.lib.sklearn import RandomForestClassifier, RandomForestRegressor
    from lale.lib.lale import ConcatFeatures
    from lale.lib.lale import Project, categorical
#    cat_columns = dataset_to_cat_columns[dataset]
#    num_columns = dataset_to_num_columns[dataset]

    try:
        (train_X, train_y), (test_X, test_y) = get_data(dataset)
        cat_columns, num_columns = get_cat_num_columns(train_X)
    except:
        cat_columns = dataset_to_cat_columns[dataset]
        num_columns = dataset_to_num_columns[dataset]
    print(cat_columns, num_columns)
    k=10#np.min([10, train_X.shape[1]])
    est = get_estimator(dataset, estimator_type, "lale")
    cat_prep = None
    num_prep = None
    if cat_columns is not None and len(cat_columns) !=0:
        cat_prep = (Project(columns=cat_columns) >> SimpleImputer(strategy="constant") 
            #>> OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            >> HashingEncoder())
    if num_columns is not None and len(num_columns) !=0:
        num_prep = (Project(columns=num_columns)
            >> SimpleImputer(strategy="mean")
            >> MinMaxScaler())

    if cat_prep is not None and num_prep is not None:
        pipeline = (cat_prep & num_prep) >> ConcatFeatures() >> Convert() >> est
    elif cat_prep is None and num_prep is not None:
        pipeline = num_prep >> SelectKBest(k=k) >> est
    elif cat_prep is not None and num_prep is None:
        pipeline = cat_prep >> SelectKBest(k=k) >> est
    return pipeline

def get_pipeline_sklearn(dataset):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_selection import SelectKBest
    from lale.lib.lale import Project, categorical
    import lale.datasets.openml
    from lale.lib.rasl.project import _columns_schema_to_list
    cat_columns, num_columns = dataset_to_cat_and_num_columns(dataset)
    if dataset in ["airlines_delay"]:
        est = RandomForestRegressor(random_state=42)
    else:
        est = RandomForestClassifier(random_state=42)
    return (#OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        make_pipeline(ColumnTransformer([("prep_cat", make_pipeline(SimpleImputer(strategy="constant"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat_columns),
                        ("prep_num", make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler()), num_columns)]),
                        SelectKBest(k=np.min([10, len(cat_columns)+len(num_columns)])),
                        est))

def get_pipeline_sklearn_large(dataset):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import SGDClassifier, SGDRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.feature_selection import SelectKBest
    from lale.lib.lale import Project, categorical
    import lale.datasets.openml
    from category_encoders.hashing import HashingEncoder
    from lale.lib.rasl.project import _columns_schema_to_list
    cat_columns, num_columns = dataset_to_cat_and_num_columns(dataset)
    if dataset in regression_datasets:
        est = SGDRegressor(random_state=42)
    else:
        est = SGDClassifier(random_state=42)
    return (#OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        make_pipeline(ColumnTransformer([("prep_cat", make_pipeline(SimpleImputer(strategy="constant"),
            #OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat_columns),
            HashingEncoder()), cat_columns),
                        ("prep_num", make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler()), num_columns)]),
                        SelectKBest(k=np.min([10, len(cat_columns)+len(num_columns)])),
                        est))

def get_pipeline_sklearn_rasl(dataset):
    from lale.lib.rasl import SimpleImputer
    from lale.lib.rasl import OrdinalEncoder
    from lale.lib.rasl import MinMaxScaler
    from lale.lib.rasl import SelectKBest
    from lale.lib.rasl import Convert
    from lale.lib.sklearn import RandomForestClassifier, RandomForestRegressor
    from lale.lib.lale import ConcatFeatures

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import make_pipeline
    from lale.lib.lale import Project, categorical
    import lale.datasets.openml
    if dataset in regression_datasets:
        (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(dataset, "regression", preprocess=False, seed=RANDOM_SEED)
    else:
        (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(dataset, "classification", preprocess=False, seed=RANDOM_SEED)
    cat_max_values = dataset_to_max_values.get(dataset, dataset_to_max_values["default"])
    cat_columns, num_columns = get_cat_num_columns(train_X)
    if dataset in ["airlines_delay"]:
        est = RandomForestRegressor(random_state=42)
    else:
        est = RandomForestClassifier(random_state=42)
    return (
        make_pipeline(ColumnTransformer([("prep_cat", make_pipeline(SimpleImputer(strategy="constant"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat_columns),
                        ("prep_num", make_pipeline(SimpleImputer(strategy="mean"), MinMaxScaler()), num_columns)]),
                        Convert(),
                        SelectKBest(k=np.min([10, train_X.shape[1]])),
                        est))

cv_types=[0,#sklearn cv
          1,#cross-validation without batching, using same-fold data for training
          2,#cross-validation with batching, using same-fold data for training
          3]#cross-validation with out-of-fold samples

n_folds=5
n_batches_per_fold=3

def get_scorer(dataset, cv_type, astype="sklearn"):
    if cv_type == 0 and dataset in classification_datasets:
        from sklearn.metrics import accuracy_score, make_scorer
        return make_scorer(accuracy_score)
    elif cv_type == 0 and dataset in regression_datasets:
        from sklearn.metrics import r2_score, make_scorer
        return make_scorer(r2_score)
    elif dataset in classification_datasets:
        import lale.lib.rasl
        accuracy_scorer = lale.lib.rasl.get_scorer("accuracy")
        return accuracy_scorer
    elif dataset in regression_datasets:
        import lale.lib.rasl
        r2_scorer = lale.lib.rasl.get_scorer("r2")
        return r2_scorer
        #from sklearn.metrics import mean_absolute_error, make_scorer
        #mae_scorer = make_scorer(mean_absolute_error)
        #return mae_scorer
