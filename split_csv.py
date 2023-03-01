import os
import pandas as pd

n_splits = 5
random_states = [42, 33, 99, 21, 60]
dataset_base_dir = "/dccstor/kakate1/rasl_datasets/"
dataset_name = "kddcup99full"
#dataset_name="ecommerce_2019_oct"
label_encoding=True
csv_path = os.path.join(dataset_base_dir, dataset_name+".csv")
#label_column = 'brand'
label_column="label"
#columns_to_drop = ['event_time', 'user_session', 'product_id']
columns_to_drop=[]

if label_encoding:
    from sklearn.preprocessing import LabelEncoder
    df = pd.read_csv(csv_path)
    y = df[label_column]
    le = LabelEncoder()
    le.fit(y)
    df[label_column]=le.transform(y)
    df.to_csv(csv_path, index=False)

for split in range(n_splits):
    output_dir = os.path.join(dataset_base_dir, "splits", "split"+str(split))
    df = pd.read_csv(csv_path)
    df = df.drop(columns_to_drop, axis=1)
    y = df[label_column]
    X = df.drop(label_column, axis=1)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_states[split])
    df = X_train
    df['label']=y_train
    df.to_csv(os.path.join(output_dir, dataset_name+"_train.csv"), index=False)
    df = X_test
    df['label']=y_test
    df.to_csv(os.path.join(output_dir, dataset_name+"_test.csv"), index=False)




