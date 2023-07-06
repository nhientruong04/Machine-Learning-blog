# (*Machine Learning*) A brief look at K-Means Clustering
## 1. Introduction
> "K-means clustering is a method of vector quantization, originally from signal processing, that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster. This results in a partitioning of the data space into Voronoi cells." - *Wikipedia*
<!-- -->
![K-means clustering example](image/k-means-example-sklearn.png)  
*Source: scikit-learn webpage*  

So, the algorithm is built with centroids, calculated by the mean distances to all of the observations in their own area, and the observations (points in 2-D), which are partitioned to the centroids base on their distances to them (belongs to centroid A if they are nearest to A compared to other centroids B, C, D). If you understand this concept, congratulation, you've just grasped the essence of the k-means clustering algorithm.  

But wait, to be more detailed, I'll show you some maths behind all of this. It would be quite intimidating at first, but it'll be fun. Believe me!  
![meme](image/believemememe.jpg)

## 2. Some Maths
### We must know what we're doing, right?
Now, let's start defining our observation and label matrix. Our observation matrix should have a form like this  
![observation_matrix](./observation_matrix.png)  
and our label matrix should have a similar form  
![label_matrix](./label_matrix.png)  
An observation will have its corresponding label vector $y$. Each element of the vector represents its binary value of that corresponding centroid. For example, suppose we have a dataset of real-estate properties having 2 features for each observation, which are the total area and prices. Then our observation matrix will have 2 rows (2 features) and N columns (depend on how large our dataset). Suppose we have magically known there are 4 groups those properties will be classified into (4 clusters), then our label matrix should have 4 rows and N columns. Now, in order to express that observation $x_1$ belongs to the $3^{rd}$ centroid, its corresponding label vector $y_1$ will have the form
$\begin{bmatrix}
0 \\
0 \\
1 \\
0 \\
\end{bmatrix}$
. Since each observation can only belong to a single centroid, if $x_1$ is classified into group $k$, then $y_{ik}=1$ and $y_{ij}=0, \forall j \neq k$. So, we've just form our conditions for the label vectors.
$$y_{ij} \in \{0, 1\}, \sum_{j=1}^K y_{ij}=1$$  

### Let's jump into a lava pool full of math notations
**Loss function**  
Let's call our centroids as $\mathbf{m}$, then we'll need to calculate the loss for each centroid. In this problem, for a 2-d dataset (points as observations), the loss for each point can be calculated as the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) from that point to its centroid. The loss from point $x_1 (a, b) \text{ to its centroid } m_k(c, d)$ is
$$
d(x_1, m_k) = \sqrt{(a - c)^2 + (b - d)^2} = \lVert \mathbf{x_1 - m_k} \rVert_2
$$
*Note: the number $_2$ is the power of the difference between two points in the square root*  
Next, we'll square it to get rid of the square root, regulize it and apply the label vector as $y_{ik}=1$, $y_{ij}=0$, $\forall j \neq k$
$$
y_{ik}\lVert \mathbf{x_i - m_k} \rVert_2^2
$$
this expression means that only when the vector label is true (has value 1), then we are able to get the loss from that point to its corresponding centroid. Otherwise, we'll still calculate the distance $d(x_i, m_j)$ but since $y_{ik} = 0$, we'll get nothing but $0$. To generalize this math for the rest of the label vector, we'll have
$$
\sum_{j=1}^K{y_{ik}\lVert \mathbf{x_i - m_k} \rVert_2^2}
$$
Generalize for the whole dataset, we'll get our loss function
$$
Loss(Y, M) =
\sum_{i=1}^N{\sum_{j=1}^K{y_{ik}\lVert \mathbf{x_i - m_k} \rVert_2^2}}
$$
Now we've got a new variables $M=\begin{bmatrix}m_1 & m_2 & \cdots & m_k\end{bmatrix}$, which are the coordinates of all centroids. Since it is a loss function, we'll need to find its smallest value (optimize) and therefore lead us to solving this equation:
$$
\text{Y, M} = argmin_{Y,M}\sum_{i=1}^N{\sum_{j=1}^K{y_{ij}\lVert \mathbf{x_i - m_j} \rVert_2^2}}
$$
$$
\text{subject to: }y_{ij} \in \{0, 1\}\; \forall i,j, \sum_{j=1}^K y_{ij}=1, \forall i
$$  
*Note:* the equation above means that we'll have to find matrices Y, M such that the loss function will have its $min$ value. The $argmin$ expression means that instead of finding the usual $min$ of the equation, which is the smallest value of the loss function, we now want to find the **argument value** resulting to that $min$.  
*Another note (Why not?):* after completing the equation, this actually resembles the **Sum of squared errors (SSE)** which has a following formula
$$
SSE=\sum_{i=1}^n{(y_i-\hat{y}_i)^2}
$$
with $y_i$ as our observations and $\hat{y}_i$ as their corresponding centroids. Take a look and ponder over it for a while, you'll see they're all related. *This is actually fun. Right?*  

### Let's optimize  
In machine learning, optimizing our loss function usually comes with taking its derivative (the slope) and tweak our weights such that its derivative will move in an opposite direction to that slope. Therefore the loss function will be reduced and ultimately we'll get its global minimum (if we're lucky enough). This is called *[Gradient Descent](https://www.ibm.com/topics/gradient-descent#:~:text=Gradient%20descent%20is%20an%20optimization,each%20iteration%20of%20parameter%20updates.)*. In our problem, however, we'll use a quite similar process but we still hold the same goal, which is finding the minimum of our loss function. But it won't be easy as we're having 2 variables $Y$ and $M$. So we'll try to ignore 1 of the variables and take the derivative of the other (for me it looks like partial derivative).  
#### Ignore $M$, find $Y$  
*"Suppose we've found our centroids, let's label our observations such that the loss function achieve the $min$ value"*. For an easier grasp, labeling for **all** observations can be seen as labeling for **a single** observation.
$$
y_i = argmin_{y_i}\sum_{j=1}^K{y_{ij}\lVert \mathbf{x_i - m_j} \rVert_2^2}
$$
$$
\text{subject to: }y_{ij} \in \{0, 1\}\; \forall j, \sum_{j=1}^K y_{ij}=1
$$  
Can you see the pattern? This equation means that finding the $min$ value depends on the distance $\lVert \mathbf{x_i - m_j} \rVert_2$, since $y_{ij}$ can only be $1$ for a single centroid, otherwise it equals $0$. Consequently, this equation shows us that **the observation $x_i$ belongs to its closest centroid**.
#### Find $M$, ignore $Y$
*"Suppose we've labeled our observations, let's calculate their new centroids such that the loss function achieve the $min$ value"*. Similar to $Y$, the equation can be shortened as follow
$$
m_j = argmin_{m_j}\sum_{i=1}^N{y_{ij}\lVert \mathbf{x_i - m_j} \rVert_2^2}
$$
This equation means that we're calculating for a single centroid. Remember the condition $y_{ij} \in \{0, 1\}, \forall i,j$ ? Now we only choose one centroid, which means a single defined $j$, and choose all observations. $y_{ij}=1$ only when that label vector of $x_i$ has the right $j$. Therefore, this equation is the sum of the distance from all points in a single cluster to that cluster centroid. Now we're gonna take the derivative of this loss function just as the *Gradient Descent* I've mentioned earlier. Remember, since we're taking a partial derivative with respect to $m_j$, $y_{ij}$ will be considered as a constant.
$$
\frac{\partial l(m_j)}{\partial m_j} = \sum_{i=1}^N{2y_{ij}(x_i-m_j)\frac{\partial (x_i-m_j)}{\partial m_j}} = 2\sum_{i=1}^N{y_{ij}(m_j-x_i)}
$$ 
Solve this derivative when it equals 0, we'll get
$$
m_j\sum_{i=1}^N{y_{ij}} = \sum_{i=1}^N{y_{ij}x_i}
$$
$$
\Rightarrow m_j =\frac{\sum_{i=1}^N{y_{ij}x_i}}{\sum_{i=1}^N{y_{ij}}}
$$
Can you see the pattern? The *numerator* is the **sum of all observations** in that cluster and the *denominator* is the **total number of observations** in that same cluster. So, it is clear that the centroid $m_j$ is the **mean of all observations in cluster $j$**.  
## 3. The real algorithm with no math in it
To sum it up, our algorithm consists of the following things:  
**Input:** dataset $X$ and number of clusters $K$  
**Output:** centroids $M$ and label matrix $Y$  
1. Initialize $K$ clusters by choosing randomly $K$ observations
2. Classify our observations into their nearest cluster, receive the label matrix $Y$
3. If the label matrix stays the same as it was in the last iteration, the algorithm will stop
4. Update the centroids by calculating the mean of their cluster observations
5. Back to step 2  
<!-- -->

## 4. Apply a whole bunch of this with Python
