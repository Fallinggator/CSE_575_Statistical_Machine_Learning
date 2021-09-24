
# coding: utf-8

# In[3]:


from Precode import *
import scipy.io as sio
import scipy.spatial.distance as distance
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.cluster
data = np.load('AllSamples.npy')
X=data


# In[4]:


k1,i_point1,k2,i_point2 = initial_S1('0345') # please replace 0111 with your last four digit of your ID


# In[10]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)
# k1= 3
# i_point1= [[5.07250754,7.89834048],[ 7.72715541,7.62018213],[ 5.02471033  8.23879873]]
# k2=5
# i_point2=[[ 3.02640736  5.74083968],[ 7.94375954,8.21165063],[ 3.89523379,0.70718356],[ 3.0226944,0.86402039],[ 6.8113456,0.99804859]]


# In[6]:


def assign(D, C): # assgin based on objective function
  output=[]
  for i in range(0,len(D)):
    compare_List=[]
    for j in range(0,len(C)):
      compare_List.append( pow(((D[i][0]-C[j][0])**2)+((D[i][1]-C[j][1])**2),0.5)) # objective function
    output.append(compare_List.index(min(compare_List)))

  return output

def centroid(D, C): # Calculate centorid
  k = max(C)+1
  cluster_List=[]
  centroids=[]
  for i in range(0,k):
    cluster_List.append([])
    centroids.append([0,0])
  k0 = []
  k1 = []
  k2 = []
  k3 = []
  k0mx,k0my,k1mx,k1my,k2mx,k2my,k3mx,k3my =0,0,0,0,0,0,0,0 
  elements_Cluster=[]
  output = []
 

    
      
    
  for i in range(0,len(C)):
    cluster_List[C[i]].append(D[i])

      
 

  for k in range(0,len(cluster_List)): #find mean cluster
    cluster = cluster_List[k] 
    kmx,kmy=0,0
    for i in range(0,len(cluster)):
      kmx += cluster[i][0]
      kmy += cluster[i][1]
    centroids[k] = [kmx/float(len(cluster)),kmy/float(len(cluster))]  
  

    
 
  return centroids

# Objective Function
def ObjectiveFunc(SampleData,centroidList,clusters,meanSquaredError):
    meanSquaredError = 0
    for i in range(0,2):
        # meanError = [(SampleData[j] - centroidList[i])  for j in range(len(SampleData)) if clusters[j] == i]
        meanSqError = [((SampleData[j] - centroidList[i])**2)  for j in range(len(SampleData)) if clusters[j] == i]
        # print('Mean Error is : ', meanError[i])
        # print('Mean Sq Error is : ', meanSqError[i])
    meanSquaredError = np.sum(meanSqError)
    ObjectiveFuncList.append(meanSquaredError)
    print(ObjectiveFuncList)
    print('meanSquaredError = ', meanSquaredError)
    return meanSquaredError


# In[9]:


select_Strategy = 1
k= k2 #k value
data_Set= data
centroids= i_point2
centroid_Count=1 #counter for max distance based on previous data point (might not be used)
found = False # find the centroids flag
iteration= 0
while(found!= True ):
      K_assignment_List= assign(data_Set,centroids) #assing and find centorid mean , repeat until convergence
      c=0
      for i in range(len(centroids)):

        if((len(centroid(data_Set,K_assignment_List))!= k)): #if a cluster k is unassigned any value stop , < k(desired input)
            found=True
            break
            
        if( (centroids[i][0]) == (centroid(data_Set,K_assignment_List)[i][0]) and (centroids[i][1]) == (centroid(data_Set,K_assignment_List)[i][1])):
          c+=1
        if(c==len(centroids)):
          found=True
          #break
        
      iteration+=1
     
      centroids = centroid(data_Set,K_assignment_List)

print("Centroids:")    
print(centroids)
#print(K_Means.centroid(data_Set,K_assignment_List))
print("K_assignment_List:")

print(K_assignment_List)
#print(c)
print("Iteration:")

print(iteration)
x_List=[]
y_List=[]
for i in range(len(centroids)):
    x_List.append(centroids[i][0])
    y_List.append(centroids[i][1])
print(centroids)   
print(K_assignment_List)

