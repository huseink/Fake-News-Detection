import pandas as pd

from sklearn.model_selection import train_test_split

# Load your dataset into a pandas dataframe
df = pd.read_csv('evaluate/test.csv')

# Split the dataframe into two equal halves
df_1, df_2 = train_test_split(df, test_size=0.5, random_state=42)
# Add an empty column named 'new_column'
df_1 = df_1.assign(label=pd.Series())
df_2 = df_2.assign(label=pd.Series())


df_1 = df_1.drop(
    ['Unnamed: 0'], axis=1)
df_2 = df_2.drop(
    ['Unnamed: 0'], axis=1)

# Save the two halves to separate CSV files
df_1.to_csv('evaluate/test-first-part.csv', index=False)
df_2.to_csv('evaluate/test-second-part.csv', index=False)
