from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Load dataset
dataset = load_iris()
# Create DataFrame from data
X = pd.DataFrame(dataset.data, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
y = pd.DataFrame(dataset.target, columns=['Targets'])
# Color map for visualization
colormap = np.array(['red', 'lime', 'black'])
# Set up the figure
plt.figure(figsize=(14, 7))
# REAL PLOT
plt.subplot(1, 3, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y.Targets.values.ravel()], s=40)
plt.title('Real')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# KMeans Clustering
scaler_kmeans = preprocessing.StandardScaler()
X_scaled_kmeans = scaler_kmeans.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42)
model.fit(X_scaled_kmeans)
predY = model.labels_
plt.subplot(1, 3, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[predY], s=40)
plt.title('KMeans')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# Gaussian Mixture Model Clustering
scaler = preprocessing.StandardScaler()
xsa = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(xsa)
y_cluster_gmm = gmm.predict(xsa)
plt.subplot(1, 3, 3)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('GMM Classification')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# Show plot
plt.suptitle('Iris Dataset Clustering Comparison', fontsize=16)
plt.tight_layout()
plt.show()
