import numpy as np
import cv2
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def makeClusters():

    centers = [[0,0,0], [6,6,0],[6,-6,0],[-6,6,0],[-6,-6,0]]
    clusters=[]

    for i in range(5):
        cluster = []
        for j in range(300):
            data = []
            data.append(np.random.normal(centers[i][0], 0.5))
            data.append(np.random.normal(centers[i][1], 0.5))
            data.append(np.random.normal(centers[i][2], 0.5))
            cluster.append(data)
        clusters.append(cluster)

    return clusters

def unzipData(clusters):
    data_values = []
    for i in range (5):
        for j in range (300):
            data_values.append(clusters[i][j])
    return data_values

def kMeans(data_values):
    # convert to float
    data_values = np.float32(data_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    k = 5
    _, labels, (centers) = cv2.kmeans(data_values, k, None, criteria, 1000, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values

    # flatten the labels array
    print("labels : ")
    labels = labels.flatten()
    print(labels)

    for i in range(1500):
        #print(labels[i])
        temp = []
        np.insert(data_values,3,(labels))
    # convert all pixels to the color of the centroids
    #segmented_data = centers[labels.flatten()]

    print('labeled data : ')

    print(data_values)

    # reshape back to the original image dimension

    print(data_values)
    return data_values

def plotClusters(clustered_data):
    print('plotting data')
    v = clustered_data
    print(v)
    df = pd.DataFrame(v, columns=['Feature1', 'Feature2', 'Feature3', "Cluster"])
    print(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['Feature1'])
    y = np.array(df['Feature2'])
    z = np.array(df['Feature3'])

    ax.scatter(x, y, z, marker="s", c=df["Cluster"], s=40, cmap="RdBu")

    plt.show()

if __name__ == "__main__":
    clusters = makeClusters()
    datas = unzipData(clusters)
    clustered_data = kMeans(datas)
    plotClusters(clustered_data)




