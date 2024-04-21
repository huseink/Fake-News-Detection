import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd


df = pd.read_csv('final/tagged_ALL.csv')

tagged_data = df.dropna(subset=['label'])


# flake8: noqa


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


X_pred = tagged_data[['tweet_text_cred_score',
                      'tweet_social_cred_score', 'author_cred_score']]


predicted_labels = kmeans.predict(X_pred)
label_map = {0: 'NC', 1: 'LC',
             2: 'C', 3: 'HC'}
predicted_labels = [label_map[label] for label in predicted_labels]

# Compare predicted labels with the existing labels
tagged_data['predicted_label'] = predicted_labels

tagged_data.to_csv('predicted_labels2.csv', index=False)
