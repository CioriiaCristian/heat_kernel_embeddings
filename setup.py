from embedpy import core
from embedpy.config import Config
from itertools import combinations
from tqdm import tqdm
import numpy as np
import os

# Processed Path
proc_path = "processed\\"
# Plots path
plot_path = "plots\\"

proc_paths = [
    "processed\\sectional_curvature\\",
    "processed\\gaussian_curvature\\"
]

plot_paths = [
    
    "plots\\classical_hausdorff\\",
    "plots\\classical_hausdorff\\gaussian_curvature\\",
    "plots\\classical_hausdorff\\sectional_curvature\\",
    "plots\\modified_hausdorff\\",
    "plots\\modified_hausdorff\\gaussian_curvature\\",
    "plots\\modified_hausdorff\\sectional_curvature\\",
]

isdir = os.path.isdir(proc_path)
if isdir == False:
    os.mkdir(proc_path)
    for _path in proc_paths:
        os.mkdir(_path)

isdir = os.path.isdir(plot_path)
if isdir == False:
    os.mkdir(plot_path)
    for _path in plot_paths:
        os.mkdir(_path)

# Choose 3 random numbers, without duplicates
combs = list(combinations(np.arange(1,Config.object_number),Config.object_sample_size))
object_indeces = combs[np.random.randint(len(combs))]

print(f'The following objects have been chosen :{object_indeces}')
poses_list = list(range(Config.object_poses))[::4]
time_domain = Config.time_domain
print('Currently generating all curvature files...')
for obj in tqdm(object_indeces):
    for pose in poses_list:
        for t in time_domain:
            core.generate_curvature_files(t,obj,pose)
