import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

media_1 = [1, 1]
cov_1 = [[1, 0], [0, 1]]
data_1 = np.random.multivariate_normal(media_1, cov_1, 100)

media_2 = [7, 2]
cov_2 = [[1, 0.5], [0.5, 1]]
data_2 = np.random.multivariate_normal(media_2, cov_2, 100)

media_3 = [1, 8]
cov_3 = [[1, -0.8], [-0.8, 1]]
data_3 = np.random.multivariate_normal(media_3, cov_3, 100)

data = np.concatenate((data_1, data_2, data_3), axis=0)

plt.scatter(data[:, 0], data[:, 1])
plt.show()

sil = []
for k in range(2, 10):
  cluster = KMeans(n_clusters=k)
  cluster.fit_predict(data)
  labels = cluster.labels_
  silhouette = silhouette_score(data, labels)
  sil.append(silhouette)
  centers = cluster.cluster_centers_
  plt.scatter(data[:,0], data[:, 1], c = labels)
  plt.scatter(centers[:,0], centers[:,1], marker='X', c='red')
  plt.title(f"Silhouette Score: {silhouette_score(data,labels)}")
  plt.show()
plt.plot(range(2, 10), sil)
plt.show()
