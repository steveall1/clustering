import numpy as np
import cv2
from mpl_toolkits import mplot3d
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans

class Centroid:
    centroids = [] # centers of clustered data, 5 exist
    cluster_std = [] #std of clustered data, 0~5

def makeData(std_x,std_y,std_z):

    centers = [[0,0,0], [6,6,0],[6,-6,0],[-6,6,0],[-6,-6,0]]
    x_data=[]
    y_data=[]
    z_data=[]

    for i in range(5):
        for j in range(300):
            data = []
            x_data.append(np.random.normal(centers[i][0], std_x))
            y_data.append(np.random.normal(centers[i][1], std_y))
            z_data.append(np.random.normal(centers[i][2], std_z))


    df = pd.DataFrame({
        'x':x_data,
        'y':y_data,
        'z':z_data
    })


    return df

def cluster(df):
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df)

    labels = kmeans.predict(df)

    print(labels)

    Centroid.centroids = kmeans.cluster_centers_
    #print(Centroid.centroids)

    df.insert(3,'l',labels,True)
    print(df)

    for i in range(5):
        x_s = []
        y_s = []
        z_s = []

        x_s = df['x'].where(df['l'] == i)
        y_s = df['y'].where(df['l'] == i)
        z_s = df['z'].where(df['l'] == i)

        xStd = np.std(x_s)
        yStd = np.std(y_s)
        zStd = np.std(z_s)
        temp = []
        temp.append(xStd)
        temp.append(yStd)
        temp.append(zStd)
        Centroid.cluster_std.append(temp)

    print('std recognition test')
    print('0 : '+str(Centroid.cluster_std[0]))
    print('1 : '+str(Centroid.cluster_std[1]))
    print('2 : '+str(Centroid.cluster_std[2]))
    print('3 : '+str(Centroid.cluster_std[3]))
    print('4 : '+str(Centroid.cluster_std[4]))


    return df

def plotClustersWithoutLabel(clustered_data):
    print('plotting data')
    v = clustered_data
    print(v)
    df = pd.DataFrame(v, columns=['x', 'y', 'z'])
    print(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['x'])
    y = np.array(df['y'])
    z = np.array(df['z'])

    ax.scatter(x, y, z, marker="s", c=df['x'], s=40, cmap="Spectral")

    plt.show()


def plotClusters(clustered_data):
    print('plotting data')
    v = clustered_data
    print(v)
    df = pd.DataFrame(v, columns=['x', 'y', 'z', 'l'])
    print(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.array(df['x'])
    y = np.array(df['y'])
    z = np.array(df['z'])

    ax.scatter(x, y, z, marker="s", c=df['l'], s=40, cmap="Spectral")

    plt.show()

def testing(interval,std_x,std_y,std_z):
    #centers = np.array([[0,0,0], [6,6,0],[6,-6,0],[-6,6,0],[-6,-6,0]])
    #x_data=[]
    #y_data=[]
    #z_data=[]

    stds = Centroid.cluster_std

    test_data = []
    print(Centroid.centroids)

    centers = Centroid.centroids
    for i in range(5):
        for j in range(100):
            data = []
            #x_data.append(np.random.normal(centers[i][0], 3))
            #y_data.append(np.random.normal(centers[i][1], 3))
            #z_data.append(np.random.normal(centers[i][2], 3))
            data.append(np.random.normal(centers[i][0], std_x))
            data.append(np.random.normal(centers[i][1], std_y))
            data.append(np.random.normal(centers[i][2], std_z))
            test_data.append(data)

    count = [0,0,0,0,0]

    print('info check (size of test_data) : '+str(len(test_data)))

    print("Cluster 0 Test : ")

    for i in range(0,100):
        distances = []
        for c in range(0, 5):
            d = np.linalg.norm(test_data[i]-centers[c])
            distances.append(d)
        idx = distances.index(min(distances))
        if(distances[idx]<=interval*np.sqrt(stds[0][0]*stds[0][0]+stds[0][1]*stds[0][1]+stds[0][2]*stds[0][2])):
                count[idx]+=1

    for p in range (0,5):
        print('classifed as cluster '+str(p)+' : '+str(count[p]))

    print('classifed as nowhere : '+str(100-sum(count)))

    count = [0,0,0,0,0]

    print("Cluster 1 Test : ")

    for i in range(100,200):
        distances = []
        for c in range(0, 5):
            d = np.linalg.norm(test_data[i]-centers[c])
            distances.append(d)
        idx = distances.index(min(distances))
        if(distances[idx]<=interval*np.sqrt(stds[1][0]*stds[1][0]+stds[1][1]*stds[1][1]+stds[1][2]*stds[1][2])):
                count[idx]+=1

    for p in range (0,5):
        print('classifed as cluster '+str(p)+' : '+str(count[p]))
    print('classifed as nowhere : '+str(100-sum(count)))

    count = [0,0,0,0,0]

    print("Cluster 2 Test : ")

    for i in range(200,300):
        distances = []
        for c in range(0, 5):
            d = np.linalg.norm(test_data[i]-centers[c])
            distances.append(d)
        idx = distances.index(min(distances))
        if(distances[idx]<=interval*np.sqrt(stds[2][0]*stds[2][0]+stds[2][1]*stds[2][1]+stds[2][2]*stds[2][2])):
                count[idx]+=1

    for p in range (0,5):
        print('classifed as cluster '+str(p)+' : '+str(count[p]))
    print('classifed as nowhere : '+str(100-sum(count)))
    count = [0,0,0,0,0]

    print("Cluster 3 Test : ")

    for i in range(300,400):
        distances = []
        for c in range(0, 5):
            d = np.linalg.norm(test_data[i]-centers[c])
            distances.append(d)
        idx = distances.index(min(distances))
        if(distances[idx]<=interval*np.sqrt(stds[3][0]*stds[3][0]+stds[3][1]*stds[3][1]+stds[3][2]*stds[3][2])):
                count[idx]+=1


    for p in range (0,5):
        print('classifed as cluster '+str(p)+' : '+str(count[p]))
    print('classifed as nowhere : '+str(100-sum(count)))

    count = [0,0,0,0,0]


    print("Cluster 4 Test : ")

    for i in range(400,500):
        distances = []
        for c in range(0, 5):
            d = np.linalg.norm(test_data[i]-centers[c])
            distances.append(d)
        idx = distances.index(min(distances))
        if(distances[idx]<=interval*np.sqrt(stds[4][0]*stds[4][0]+stds[4][1]*stds[4][1]+stds[4][2]*stds[4][2])):
                count[idx]+=1

    for p in range (0,5):
        print('classifed as cluster '+str(p)+' : '+str(count[p]))
    print('classifed as nowhere : '+str(100-sum(count)))

    #centers = np.array([[0,0,0], [6,6,0],[6,-6,0],[-6,6,0],[-6,-6,0]])
    free_center = np.array([10,10,10])
    free_data = []
    for j in range(100):
        data = []
        data.append(np.random.normal(free_center[0], std_x))
        data.append(np.random.normal(free_center[1], std_y))
        data.append(np.random.normal(free_center[2], std_z))
        free_data.append(data)


    count = [0,0,0,0,0]


    print("Free Test : ")

    for i in range(0,100):
        distances = []
        for c in range(0, 5):
            d = np.linalg.norm(free_data[i]-centers[c])
            distances.append(d)
        idx = distances.index(min(distances))
        if(distances[idx]<=interval*np.sqrt(std_x*std_x+std_y*std_y+std_z*std_z)):
                count[idx]+=1

    for p in range (0,5):
        print('classifed as cluster '+str(p)+' : '+str(count[p]))
    print('classifed as nowhere : '+str(100-sum(count)))







if __name__ == "__main__":
    df = makeData(2,3,2)
    #parameter : std_x,std_y,std_z
    plotClustersWithoutLabel(df)
    clustered_data = cluster(df)
    plotClusters(clustered_data)
    testing(1.5,2,3,2)
    #parameter : cf_interval, std