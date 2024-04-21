# flake8: noqa

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('final/stripped_merged_data.csv')

# Select the columns to cluster
X = df[['tweet_text_cred_score', 'tweet_social_cred_score',
        'author_cred_score']]

# Fit KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

# Get the labels for each row
labels = kmeans.labels_

# Map the labels to string values
label_map = {0: 'Not credible', 1: 'Low credibility',
             2: 'Credible', 3: 'High credibility'}
labels = [label_map[label] for label in labels]

# Add the labels to the DataFrame
df['label'] = labels

# Save the labeled data to a new CSV file
# df.to_csv('final/labeled_without_overall_dataset.csv', index=False)

# # Perform PCA to reduce the data to 2 dimensions
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Plot the clusters
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('KMeans Clustering with 4 clusters')
# plt.show()


# # Predict the label for a new tweet with scores of 0.7, 0.8, and 0.9
# new_tweet_scores = [4.3, 0.14, 51.6, 18.5]
# new_tweet_scores_df = pd.DataFrame([new_tweet_scores], columns=X.columns)
# label = kmeans.predict(new_tweet_scores_df)

# # Map the integer label to a string label
# label_map = {0: 'Not credible', 1: 'Low credibility',
#              2: 'Credible', 3: 'High credibility'}
# predicted_label = label_map[label[0]]

# # Print the predicted label
# print('The predicted label for the new tweet is:', predicted_label)
