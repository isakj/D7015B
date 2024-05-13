# %%
#%%
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import statistics
import math
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# %%
#%% utility functions
def show_cloud(points_plt):
    ax = plt.axes(projection='3d')
    ax.scatter(points_plt[:,0], points_plt[:,1], points_plt[:,2], s=0.01)
    plt.show()

def show_scatter(x,y):
    plt.scatter(x, y)
    plt.show()

def get_ground_level(pcd):
    resolution = 0.1 # meter
    min = math.floor(np.min(pcd[:,2]) / resolution)
    max = math.ceil(np.max(pcd[:,2]) / resolution)

    cnts, bins = np.histogram(pcd[:,2], bins=max-min, range=(min*resolution,max*resolution))

    if False:
        threshold = statistics.median(cnts)
        found_peak = False

        for count,height in zip(cnts,bins):
            if not found_peak:
                found_peak = count > threshold*3
            else:
                if count < threshold * 2:
                    ground = height + resolution/2
                    break

    argmax = np.argmax(cnts)

    cnts = cnts[argmax:]
    bins = bins[argmax:]

    queue = []
    found_peak = False
    for count,height in zip(cnts,bins):
        l = len(queue)
        if l > 3:
            if np.sum(queue[0:l//2]) < np.sum(queue[l//2:l]) * 0.7:
                ground = height + resolution/2
                break
            queue = queue[1:]
        queue.append(count)

    return ground

def cluster_and_bbox(pcd, eps):
    # Calculate clustering
    # 
    # Parameters: 
    #   pcd: point-cloud data
    #   eps: epsilot
    #
    # Returns:
    #   tuple of array of 8 elements and the cluster labels
    #     Array contains epsilon, size of largest cluster, xmin_bbox, xmax_bbox, ymin_bbox, ymax_bbox, cluster index
    #

    clustering = DBSCAN(eps = eps, min_samples=5).fit(pcd)
    sizes = np.empty((0,6))

    # loop to find sizes
    for i in np.unique(clustering.labels_[clustering.labels_ >= 0]):
        points = pcd[clustering.labels_ == i]
        sizes = np.append(sizes, [[i, (np.max(points[:,0])-np.min(points[:,0]))*(np.max(points[:,1])-np.min(points[:,1])), 
            np.min(points[:,0]), np.max(points[:,0]), np.min(points[:,1]), np.max(points[:,1])]], axis=0)

    # sort according to bbox size
    sorted_indices = np.argsort(-sizes[:,1])
    sizes = sizes[sorted_indices]
    sizes = sizes[0]
    points_in_bbox = (pcd[:,0] >= sizes[2]) & (pcd[:,0] <= sizes[3]) & \
        (pcd[:,1] >= sizes[4]) & (pcd[:,1] <= sizes[5])
    points_in_bbox_and_cluster = points_in_bbox & (clustering.labels_ == sizes[0])
    data = [eps, sizes[1], np.count_nonzero(points_in_bbox_and_cluster) / np.count_nonzero(points_in_bbox)]
    data = np.append(data, sizes[2:6])
    data = np.append(data, [sizes[0]])
    
    return (data,clustering.labels_)

def best_epsilon(pcd):
    nearest_neighbors = NearestNeighbors(n_neighbors=30)
    neighbors = nearest_neighbors.fit(pcd)

    distances, indices = neighbors.kneighbors(pcd)
    distances = np.sort(distances[:,29], axis=0)

    i = np.arange(len(distances))
    knee = KneeLocator(i, distances, curve='convex', direction='increasing', interp_method='polynomial')

    return distances[knee.knee]


# %%
#%% read file containing point cloud data
pcd = np.load("dataset1.npy")

pcd.shape


# %%
#%% show downsampled data in external window
%matplotlib qt
show_cloud(pcd)
#show_cloud(pcd[::10]) # keep every 10th point


# %%
#%% remove ground plane

'''
Task 1 (3)
find the best value for the ground level
One way to do it is useing a histogram 
np.histogram

update the function get_ground_level() with your changes

For both the datasets
Report the ground level in the readme file in your github project
Add the histogram plots to your project readme
'''
min = math.floor(np.min(pcd[:,2]) / resolution)
max = math.ceil(np.max(pcd[:,2]) / resolution)
cnts, bins = np.histogram(pcd[:,2], bins=max-min, range=(min*resolution,max*resolution))
plt.bar(bins[:-1] + np.diff(bins) / 2, cnts, np.diff(bins))
#plt.title('Histogram, number of points per interval (bin size=0.1m)')
plt.xlabel('height (z)')
plt.ylabel('number of points')


est_ground_level = get_ground_level(pcd)
print(est_ground_level)

pcd_above_ground = pcd[pcd[:,2] > est_ground_level] 


# %%
#%% side view
show_cloud(pcd_above_ground)



# %%
# %%
# Plotting resulting clusters

print(clustering.labels_)
ax = plt.axes(projection='3d')
ax.scatter(pcd_above_ground[:,0], pcd_above_ground[:,1], pcd_above_ground[:,2], c=clustering.labels_, s=0.01)
plt.show()



# %%
def k_distances4(x):
    l=x.shape[0]
    sum_matrix=np.eye(x.shape[0], x.shape[0])*10000000
    for dim in range(0,x.shape[1]):
        for l in range(0,x.shape[0]):
            distance_vector = np.subtract(x[l,dim]*np.ones((x.shape[0],1)),x[:,dim].reshape(x.shape[0],1))
            distance_vector = np.square(distance_vector)
            distance_vector = np.add(sum_matrix[:,l].reshape(x.shape[0],1), distance_vector)
            sum_matrix[:,l] = distance_vector.flatten()
    return np.sort(np.sort(sum_matrix,axis=1)[:,0])

distance_matrix = k_distances4(pcd_above_ground)
distance_matrix = pd.Series(np.sqrt(distance_matrix))

# histogram on linear scale
hist, bins = np.histogram(distance_matrix, bins=20)

# histogram on log scale. 
# Use non-equal bin sizes, such that they look equal on log scale.
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(distance_matrix, bins=logbins,log=True)
plt.xscale('log')
plt.xlabel('distance')
plt.ylabel('number of points')
plt.show()


# %%
#%%
'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''





# %%
# %%
EPS = np.empty((0,2))
for unoptimal_eps in np.logspace(-1,1,num=100):
    # find the elbow
    (data,labels) = cluster_and_bbox(pcd_above_ground, unoptimal_eps)
    cl = len(set(labels))
    print(unoptimal_eps,cl)
    EPS = np.append(EPS, [[unoptimal_eps, cl]], axis=0)

plt.plot(EPS[:,0],EPS[:,1])
plt.yscale('log')
plt.xscale('log')
plt.xlabel('$\epsilon$')
plt.ylabel('number of clusters')


# %%
df = pcd_above_ground
nearest_neighbors = NearestNeighbors(n_neighbors=2)
neighbors = nearest_neighbors.fit(df)

distances, indices = neighbors.kneighbors(df)
distances = np.sort(distances[:,1], axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.yscale('log')
plt.xscale('log')

hist, bins = np.histogram(distances, bins=20)

# histogram on log scale. 
# Use non-equal bin sizes, such that they look equal on log scale.
plt.figure()
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(x, bins=logbins,log=True)
plt.xscale('log')
plt.show()

from kneed import KneeLocator

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing')

plt.figure()
knee.plot_knee()
plt.xscale('log')
plt.yscale('log')
plt.xlabel("cumulative number of points")
plt.ylabel("distance")
plt.figure()
knee.plot_knee()
#plt.xscale('log')
#plt.yscale('log')
plt.xlim(63000,65000)
plt.xlabel("cumulative number of points")
plt.ylabel("distance")
#knee.plot_knee_normalized()
#plt.yscale('log')

print(knee.knee)
print(distances[knee.knee])

# %%
#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''


# %%
# %%
# Plotting resulting clusters
(data,labels) = cluster_and_bbox(pcd_above_ground, best_epsilon(pcd_above_ground))
#(data,labels) = cluster_and_bbox(pcd_above_ground, 0.7)

# labels start at -1 (for noise). We move it to 0, so that colormap works
labels=labels+1
clusters = len(set(labels)) - (1 if -1 in labels else 0)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, clusters)]
best = np.int64(data[7])+1
match = colors[best]
colors = [(1-(1-r)/3,1-(1-g)/3,1-(1-b)/3,a) for (r,g,b,a) in colors]
colors[best] = match

plt.figure()
plt.scatter(pcd_above_ground[:,0], 
            pcd_above_ground[:,1],
            c=labels,
            cmap=matplotlib.colors.ListedColormap(colors),
            s=2)


#plt.title('DBSCAN: %d clusters' % len(np.unique(labels)))
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.gca().axis('equal')

bbox = data[3:7]
print('bounding box',bbox)

rect = plt.Rectangle((bbox[0], bbox[2]), bbox[1]-bbox[0], bbox[3]-bbox[2], fill=False)
plt.gca().add_patch(rect)

plt.show()


# %%
EPS = np.empty((0,7))
for unoptimal_eps in np.logspace(-1,1,num=1000):
    (data,labels) = cluster_and_bbox(pcd_above_ground, unoptimal_eps)
    EPS = np.append(EPS, [data], axis=0)


fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(EPS[:,0],EPS[:,1],color=color)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('$\epsilon$')
ax1.set_ylabel('Size of bounding box on ground m$^2$', color=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(EPS[:,0],EPS[:,2]*100,color=color)
ax2.set_ylabel('Percentage corresponding to largest cluster', color=color)



