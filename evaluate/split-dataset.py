# flake8: noqa

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Load the dataset from CSV
df = pd.read_csv("final/merged_data.csv")

# Split the dataset into 90% train, 10% test using GroupShuffleSplit
# replace "group_col" with the name of your group column
# groups = df["username"]
# gss = GroupShuffleSplit(n_splits=1, test_size=0.1,
#                         random_state=42, stratified=False)
# train_idx, test_idx = next(gss.split(df, groups=groups))

# train_df = df.iloc[train_idx]
# test_df = df.iloc[test_idx]

# Split the dataset into 90% train, 10% test while preserving the groups
training = df.groupby('username').apply(lambda x: x.sample(frac=0.9))

testing = df.loc[set(df.index) - set(training.index.get_level_values(1))]


train_df = training.drop(
    ['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
test_df = testing.drop(
    ['Unnamed: 0', 'Unnamed: 0.1'], axis=1)

print(len(train_df))
print(len(test_df))
train_df.to_csv('evaluate/train.csv')
test_df.to_csv('evaluate/test.csv')

# Continue with the rest of the preprocessing and modeling steps
