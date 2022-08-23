from embedpy.config import Config
from scipy.spatial import Delaunay
from itertools import permutations, combinations, product
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import scipy
import cv2 as cv
import pandas as pd
import os


def image_name_format(object_number: int, object_pose: int):
    return f"obj{object_number}__{object_pose}.png"


def find_feature_points(img_filename: str):

    img = cv.imread(img_filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=Config.features_maxCorners,
        qualityLevel=Config.features_qualityLevel,
        minDistance=Config.features_minDistance,
    )
    corners = np.int0(corners)
    return np.array([corners[i].ravel() for i in range(corners.shape[0])])


def compute_adjacency_matrix(points: np.array):

    tri = Delaunay(points)
    adjacency = np.zeros((tri.points.shape[0], tri.points.shape[0]))
    for simplex in tri.simplices:
        vertex_pairs = list(permutations(simplex, 2))  # Vertex pair that have an edge
        for pair in vertex_pairs:
            adjacency[pair[0], pair[1]] = 1
    return adjacency


def compute_laplacian(adjacency_matrix: np.array):
    degree_matrix = np.diag(np.sum(adjacency_matrix, -1))
    return degree_matrix - adjacency_matrix


def compute_normalized_laplacian(adjacency_matrix: np.array):

    degree_matrix = np.diag(np.sum(adjacency_matrix, -1))
    _inv_degree_matrix = np.linalg.inv(degree_matrix)
    laplacian = compute_laplacian(adjacency_matrix)
    return np.matmul(
        np.matmul(np.sqrt(_inv_degree_matrix), laplacian), np.sqrt(_inv_degree_matrix)
    )


def euclidean_distance_matrix(
    vertex_index_1, vertex_index_2, time, eigvalues, eigvecmatrix
):
    _sum = 0
    for _vertex_index in range(eigvalues.shape[0]):
        _sum += np.exp(-eigvalues[_vertex_index] * time) * (
            (
                eigvecmatrix[_vertex_index, vertex_index_1]
                - eigvecmatrix[_vertex_index, vertex_index_2]
            )
            ** 2
        )
    # The factor of 2 is there to account double summing
    _sum = _sum /2
    # This step is necessary because sometimes lack of accuracy on last decimal will make the distance 
    #greater than 1
    if _sum>1.0:
        _sum = 1.0
    return np.sqrt(_sum)


def sectional_curvature(vertex_index_1, vertex_index_2, time, eigvalues, eigvecmatrix):

    euclidean_distance = euclidean_distance_matrix(
        vertex_index_1, vertex_index_2, time, eigvalues, eigvecmatrix
    )
    return np.sqrt(24 * (1 - euclidean_distance))


def gaussian_curvature(triangle, points, time, eigvalues, eigvecmatrix):

    euclidean_distances_list = []
    sectional_radii_list = []
    for pair in list(combinations(triangle, 2)):

        euclidean_distance = euclidean_distance_matrix(
            pair[0], pair[1], time, eigvalues, eigvecmatrix
        )
        euclidean_distances_list.append(euclidean_distance)
        sec_curvature = sectional_curvature(
            pair[0], pair[1], time, eigvalues, eigvecmatrix
        )
        sectional_radii_list.append(sec_curvature)

    avg_sectional_radius = np.mean(np.array(sectional_radii_list))
    avg_square_euclidean_distance = np.mean(np.array(euclidean_distances_list) ** 2)
    total_internal_angle = np.sum(1 / np.array(sectional_radii_list)) / 2

    return avg_square_euclidean_distance / (
        2 * avg_sectional_radius**3 * (total_internal_angle - np.pi)
    )


def generate_curvature_files(t, obj, pose):

    identifier_string = (
        "id_" + str(obj).rjust(2, "0") + str(pose).rjust(2, "0") + str(t).rjust(5, "0")
    )

    image_filename = Config.data_location + image_name_format(obj, pose)
    
    points = find_feature_points(image_filename)

    tri = Delaunay(points)

    adjancecy_matrix = compute_adjacency_matrix(points=points)

    normalized_laplacian = compute_normalized_laplacian(
        adjacency_matrix=adjancecy_matrix
    )

    (eigvalues, eigvecmatrix) = np.linalg.eig(normalized_laplacian)

    sectional_curvatures = np.zeros((len(tri.points), len(tri.points)))
    for simplex in tri.simplices:
        vertex_pairs = list(combinations(simplex, 2))  # Vertex pair that have an edge
        for pair in vertex_pairs:
            pair = list(np.sort(pair))
            _sec_curv = sectional_curvature(
                pair[0], pair[1], t, eigvalues, eigvecmatrix
            )
            sectional_curvatures[pair[0], pair[1]] = _sec_curv

    # Eliminate double counting
    export_sectional_curvatures = []
    for pair in list(combinations(range(len(tri.points)), 2)):
        _curv = sectional_curvatures[pair[0], pair[1]]
        if _curv > 0:
            export_sectional_curvatures.append([pair, _curv, obj, pose, t])

    export_sectional_curvatures = np.array(export_sectional_curvatures,dtype=object)

    gaussian_curvatures = []
    for idx, simplex in enumerate(tri.simplices):
        _gauss_curv = gaussian_curvature(simplex, points, t, eigvalues, eigvecmatrix)
        gaussian_curvatures.append([idx, _gauss_curv, obj, pose, t])

    gaussian_curvatures = np.array(gaussian_curvatures)

    df = pd.DataFrame(
        {
            "pair": export_sectional_curvatures[:, 0],
            "sectional_curvature": export_sectional_curvatures[:, 1],
            "object_number": export_sectional_curvatures[:, 2],
            "pose_number": export_sectional_curvatures[:, 3],
            "time": export_sectional_curvatures[:, 4],
            "identifier": [
                identifier_string for i in range(export_sectional_curvatures.shape[0])
            ],
        }
    )
    df.to_csv(
        "processed//sectional_curvature//obj_"
        + str(obj).rjust(2, "0")
        + "_pose_"
        + str(pose).rjust(2, "0")
        + "_time_"
        + str(t).rjust(5, "0")
        + ".csv"
    )
    df = pd.DataFrame(
        {
            "triangle_index": gaussian_curvatures[:, 0],
            "gaussian_curvature": gaussian_curvatures[:, 1],
            "object_number": gaussian_curvatures[:, 2],
            "pose_number": gaussian_curvatures[:, 3],
            "time": gaussian_curvatures[:, 4],
            "identifier": [
                identifier_string for i in range(gaussian_curvatures.shape[0])
            ],
        }
    )
    df.to_csv(
        "processed//gaussian_curvature//obj_"
        + str(obj).rjust(2, "0")
        + "_pose_"
        + str(pose).rjust(2, "0")
        + "_time_"
        + str(t).rjust(5, "0")
        + ".csv"
    )


def compute_hausdorff_distance(curvatures0, curvatures1):

    length0 = len(curvatures0)
    length1 = len(curvatures1)
    index_pairs = product(range(length0),range(length1))
    directed_distance = np.zeros((length0,length1))
    for pair in index_pairs:
        directed_distance[pair[0],pair[1]] = np.abs(curvatures0[pair[0]] - curvatures1[pair[1]])
    # The min of each column after which a max is applied aka graph1 to graph2 distance
    _dist1 = np.max(np.min(directed_distance,axis=0))
    # The min of each row after which a max is applied aka graph2 to graph1 distance
    _dist2 = np.max(np.min(directed_distance,axis=1))
    # Hausdorff distance is thus

    return max(_dist1,_dist2)

def compute_modified_hausdorff_distance(curvatures0, curvatures1):

    length0 = len(curvatures0)
    length1 = len(curvatures1)
    index_pairs = product(range(length0),range(length1))
    directed_distance = np.zeros((length0,length1))
    for pair in index_pairs:
        directed_distance[pair[0],pair[1]] = np.abs(curvatures0[pair[0]] - curvatures1[pair[1]])
    # The min of each column after which a max is applied aka graph1 to graph2 distance
    _dist1 = np.mean(np.min(directed_distance,axis=0))
    # The min of each row after which a max is applied aka graph2 to graph1 distance
    _dist2 = np.mean(np.min(directed_distance,axis=1))
    # Hausdorff distance is thus

    return max(_dist1,_dist2)


def generate_distance_matrix(time, feature_type, metric_type):
    
    if feature_type == 'sectional_curvature':
        directory_name = os.getcwd() + '\\processed\\sectional_curvature\\'
    else:
        directory_name = os.getcwd() + '\\processed\\gaussian_curvature\\'
    
    file_names = os.listdir(directory_name)
    filtered_names = [name for name in file_names if name[-9:-4]==str(time).rjust(5, "0")]
    
    distance_matrix = np.zeros((len(filtered_names), len(filtered_names)))
    
    if metric_type == 'classical_hausdorff':
        distance_function = compute_hausdorff_distance
    else:
        distance_function = compute_modified_hausdorff_distance

    print(f'Computing the distance matrix for all the graphs at time = {time} for feature = {feature_type} with distance function = {metric_type}')
    for pair in tqdm(list(combinations(range(len(filtered_names)),2))):
        curvatures0 = pd.read_csv(directory_name + filtered_names[pair[0]])[feature_type].values.tolist()
        curvatures1 = pd.read_csv(directory_name + filtered_names[pair[1]])[feature_type].values.tolist()
        
        distance_matrix[pair[0],pair[1]] = distance_function(curvatures0, curvatures1)
        distance_matrix[pair[1],pair[0]] = distance_matrix[pair[0],pair[1]]
    
    return distance_matrix

def generate_multiscaling_matrix(distance_matrix):
    
    index_pairs = list(product(range(distance_matrix.shape[0]),range(distance_matrix.shape[0])))
    t_matrix = np.zeros(distance_matrix.shape)
    for pair in index_pairs:
        row_averaged = np.mean(distance_matrix,axis = 1)
        column_averaged = np.mean(distance_matrix, axis = 0)
        matrix_averaged = np.mean(distance_matrix)
        #print(row_averaged[pair[0]], column_averaged[pair[1]],matrix_averaged)
        t_matrix[pair[0],pair[1]] = -.5*(distance_matrix[pair[0],pair[1]]**2 - row_averaged[pair[0]]**2 - column_averaged[pair[1]]*2 + matrix_averaged**2)
        t_matrix[pair[1],pair[0]] = t_matrix[pair[0],pair[1]]
        
    (eigvalues, eigvecmatrix) = np.linalg.eig(t_matrix)
    
    # Choose only the 2 dominant eigenvectors
    arg_sort_list = np.argsort(eigvalues)
    max_eigvector = eigvecmatrix[:,arg_sort_list[-1]]*np.sqrt(eigvalues[arg_sort_list[-1]])
    second_max_eigvector =eigvecmatrix[:,arg_sort_list[-2]]*np.sqrt(eigvalues[arg_sort_list[-2]])
    x_embedding = np.array([max_eigvector.tolist(),second_max_eigvector.tolist()]).T
    
    return x_embedding

def save_multiscaling_embedding_figure(x_embedding, time, feature_type, metric_type):
    
    directory_name = os.getcwd() + f'\\plots\\{metric_type}\\{feature_type}\\'
    fig = plt.figure()
    poses_used_no = Config.object_poses_used
    # Plotting 18 poses per object. The graphs are all in order so the first 18 graphs belong to the first object etc.
    plt.scatter(x_embedding[:,0][:poses_used_no], x_embedding[:,1][:poses_used_no])
    plt.scatter(x_embedding[:,0][poses_used_no:poses_used_no*2], x_embedding[:,1][poses_used_no:poses_used_no*2])
    plt.scatter(x_embedding[:,0][poses_used_no*2:poses_used_no*3],x_embedding[:,1][poses_used_no*2:poses_used_no*3])
    plt.savefig(directory_name + 'fig_time_'+str(time).rjust(5,'0')+'.png')