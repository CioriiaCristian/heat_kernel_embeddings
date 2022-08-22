from embedpy import core
from embedpy.config import Config
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

object_indeces = [np.random.randint(1,Config.object_number) for i in range(Config.object_sample_size)]
print(f'The following objects have been chosen :{object_indeces}')
poses_list = list(range(Config.object_poses))[::4]
time_domain = Config.time_domain
for obj in object_indeces:
    for pose in poses_list:
        for t in time_domain:
            core.generate_curvature_files(t,obj,pose)
