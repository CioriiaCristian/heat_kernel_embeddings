class Config:

    data_location = r'C://Users//Cristian//Documents//gitlab//complex_networks//coil-20-proc//'
    object_number = 20
    object_poses = 72
    object_poses_used = 18
    object_sample_size = 3
    time_domain = [0.01,0.1,1,10]

    # Features algorithm parameters
    features_maxCorners=20
    features_qualityLevel=0.001
    features_minDistance=20
    # These were found by trial and error to work best
    ##################################
    
    metric_type_index = {'sectional_curvature':'pair', 'gaussian_curvature':'triangle'}
    feature_types = ['sectional_curvature','gaussian_curvature']
    metric_types = ['classical_hausdorff','modified_hausdorff']
