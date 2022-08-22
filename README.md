# embedpy

Embedpy is a python library developed with the goal of showing the efficiency of differential geometry based clustering algorithm, in particular, multidimensional scaling. 

The heat kernel embedding algorithm is applied on the [COIL-100 Dataset](https://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php). 

As the very first step, download the COIL-100 database and change data_location parameter inside the "embedpy\\config.py" to the local database location.

When in the heat_kernel_embeddings project directory, open the terminal and run:

## Installation
First, set up the Python virtual environment and install all dependencies:

```shell
$   pip install -r requirements
$   python3 setup.py
```
Then run the program via

```shell
$   python3 run.py 
```

The project was made so as to reproduce the results found in [Heat Kernel Embeddings,Differential Geometry and Graph Structure](https://www.researchgate.net/publication/281978933_Heat_Kernel_Embeddings_Differential_Geometry_and_Graph_Structure) by _H. Ghawalby et al_ . 


The algorithm is the following:
1. 18 equally spaced poses of 3 randomly chosen objects of the dataset are chosen
2. Feature points are being extracted out of each image in particular
3. Delaunay triangulations are computed based on the sets of feature points
4. Sectional and gaussian curvatures are computed based on the edges and triangles of the Delaunay representations
5. Distances between the graphs are computed using both Classical Hausdorff distance and Modified Hausdorff Distance.
6. Multidimensional scaling is applied on the distance matrices to reveal the separation between the graphs

As can be seen, the Modified Hausdorff based multidimensional scaling separated the most clearly the clusters representing the poses of each object.

The _run.py_ file will generate all the results of the paper and save them in the **plots** folder. 

