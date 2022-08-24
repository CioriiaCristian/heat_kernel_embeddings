from embedpy import core
from embedpy.config import Config
import matplotlib.pyplot as plt
import pandas as pd

rand_index_labels = []
rand_index_values = []
for metric_type in Config.metric_types:
    for feature_type in Config.feature_types:
        rand_index_labels.append(metric_type + '  ' + feature_type)
        _rand = []
        for time in Config.time_domain:
            distance_matrix = core.generate_distance_matrix(time = time, feature_type= feature_type, metric_type = metric_type)
            x_embedding = core.generate_multiscaling_matrix(distance_matrix)
            core.save_multiscaling_embedding_figure(x_embedding, time, feature_type, metric_type)
            _rand.append(core.compute_rand_index(x_embedding))
        rand_index_values.append(_rand)

df = pd.DataFrame(rand_index_values, columns = ['t=0.01','t=0.1','t=1','t=10'], index= rand_index_labels)
df.to_excel('rand_table.xlsx')
