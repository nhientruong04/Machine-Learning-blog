# (*Machine Learning*) A brief look at K-Means Clustering
## 1. Introduction
> "K-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells." - *Wikipedia*
<!-- -->
![K-means clustering example](image/k-means-example-sklearn.png)  
*Source: scikit-learn webpage*  

So, the algorithm is built with centroids, calculated by the mean distances to all of the observations in their own area, and the observations (points in 2-D), which are partitioned to the centroids base on their distances to them (belongs to centroid A if they are nearest to A compared to other centroids B, C, D). If you understand this concept, congratulation, you've just grasped the essence of the k-means clustering algorithm.  
<!-- unsupervised -->
 
## 2. The real algorithm
To sum it up, our algorithm consists of the following things:  
**Input:** dataset *X* and number of clusters *K*  
**Output:** centroids *M* and label matrix *Y*
1. Initialize *K* clusters by choosing randomly *K* observations
2. Classify our observations into their nearest cluster, receive the label matrix *Y*
3. If the label matrix stays the same as it was in the last iteration, the algorithm will stop
4. Update the centroids by calculating the mean of their cluster observations
5. Back to step 2  
<!-- -->

## 3. Apply with Python
