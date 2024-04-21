from sklearn.manifold import TSNE
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


X_pred = tagged_data[['tweet_text_cred_score',
                      'tweet_social_cred_score', 'author_cred_score']]


predicted_labels = kmeans.predict(X_pred)
label_map = {0: 'NC', 1: 'LC',
             2: 'C', 3: 'HC'}
predicted_labels = [label_map[label] for label in predicted_labels]

# Compare predicted labels with the existing labels
tagged_data['predicted_label'] = predicted_labels

tagged_data.to_csv('predicted_labels4.csv', index=False)


cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Map the labels to "NC", "C", "LC", "HC"
label_map = {0: 'NC', 1: 'C', 2: 'LC', 3: 'HC'}
labels = [label_map[label] for label in labels]

# Add the labels to the DataFrame
df['label'] = labels

# Apply t-SNE for dimensionality reduction to 2D
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)

# Plot the clusters in 2D
plt.figure(figsize=(10, 8))
for label in set(labels):
    cluster_points = X_tsne[df['label'] == label]
    plt.scatter(cluster_points[:, 0],
                cluster_points[:, 1], label=f'{label} Kümesi')

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('KMeans kümelerinin t-SNE görüntüsü')
plt.legend()
plt.show()
