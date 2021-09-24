
# coding: utf-8

# In[1]:


from Precode2 import *
import scipy.io
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.cluster
data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S2('0343') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)
# k1=4
# i_point1=[2.06136024,3.54047797]
# k2=6
# i_point2=[4.9511002,8.08344216]


# In[4]:


Samp_X = []
Samp_Y = []

for sample in data:
    Samp_X.append(sample[0])
    Samp_Y.append(sample[1])
ObjFunc1 = []
ObjFunc2 = []
for x in range(0, 2):

    K_Values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # implement for k from 2 to 10
    for k in range(2, 11):
        print("K value:", k)
        centroids_array = [i_point2]
#         # Select the first centroid randomly from data.
#         indices = np.random.choice(300, size=1, replace=False, p=None)
#         print('indices=')
#         print(indices)
#         for index in indices:
#             centroids_array.append((data[index][0], data[index][1]))

        for i in range(1, k):
            strategy2 = []
            for sample in data:
                dista = 0
                for center in centroids_array:
                    dista = dista + distance.euclidean(sample, center)

                avg_distance = dista / len(centroids_array)
                strategy2.append(avg_distance)

            max_dist_index = strategy2.index(max(strategy2))
            centroids_array.append(data[max_dist_index])
        print("")
        print("Initial Cluster Centers:", centroids_array)
        print("")

        final_centroids = []
        final_cluster_matrix = []
        Stop_Cond = False
        while (Stop_Cond != True):
            # create a distances matrix
            distances = np.ndarray(shape=(300, k), dtype=float)

            # To compute distance from each cluster center to each data point.

            for i in range(len(data)):
                for j in range(k):
                    dist = distance.euclidean(data[i], centroids_array[j])
                    distances[i][j] = dist        
            # Assign datapoints to the nearest centroid.

            c_index = []

            for i in range(len(distances)):
                distance_list = distances[i].tolist()
                min_index = distance_list.index(min(distance_list))
                c_index.append(min_index)

            # Recalculating the cluster centroids.
            new_cluster_centroids = []

            cluster_matrix = []
            for i in range(k):
                cluster = []
                new_centre = (0, 0)
                xsum = 0
                ysum = 0

                for j in range(len(c_index)):
                    if (i == c_index[j]):
                        cluster.append(data[j])

                for l in range(len(cluster)):
                    xsum = xsum + cluster[l][0]
                    ysum = ysum + cluster[l][1]
                if len(cluster) != 0:
                    new_centre = (xsum / len(cluster), ysum / len(cluster))
                    new_cluster_centroids.append(new_centre)
                    cluster_matrix.append(cluster)
                else:
                    new_cluster_centroids.append(centroids_array[i])


            # stop if new cluster centroids are equal to previous ones.

            count = 0

            for i in range(k):
                if (new_cluster_centroids[i][0] == centroids_array[i][0] and new_cluster_centroids[i][1] == centroids_array[i][1]):
                    count = count + 1

            if (count == k):
                Stop_Cond = True
                final_centroids = new_cluster_centroids
                final_cluster_matrix = cluster_matrix
 
            if Stop_Cond == False:
                for i in range(k):
                    centroids_array[i] = new_cluster_centroids[i]


        print("Final Cluster Centers:", final_centroids)
        print("")


        # Calculating the Objective Function value
        obj_value = 0.0
        for i in range(k):
            current_center = final_centroids[i]
            for sample in final_cluster_matrix[i]:
                obj_value = obj_value + (
                            distance.euclidean(sample, current_center) * distance.euclidean(sample, current_center))
        if (x == 0):
            ObjFunc1.append(obj_value)
        else:
            ObjFunc2.append(obj_value)

# Plotting Objective Function vs k values
print("First Initialization")
plt.plot(K_Values, ObjFunc1)
plt.title('Elbow Graph')
plt.xlabel('No of Clusters')
plt.ylabel('Objective Function')
plt.show()

print("Second Initialization")
plt.plot(K_Values, ObjFunc2)
plt.title('Elbow Graph')
plt.xlabel('No of Clusters')
plt.ylabel('Objective Function')
plt.show()

