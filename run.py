from embedpy import core
from embedpy.config import Config

for time in Config.time_domain:
    for feature_type in Config.feature_types:
        for metric_type in Config.metric_types:

            distance_matrix = core.generate_distance_matrix(time = time, feature_type= feature_type, metric_type = metric_type)
            x_embedding = core.generate_multiscaling_matrix(distance_matrix)
            core.save_multiscaling_embedding_figure(x_embedding, time, feature_type, metric_type)