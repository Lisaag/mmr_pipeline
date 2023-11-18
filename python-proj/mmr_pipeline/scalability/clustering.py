import scalability.tsne as tsne
import pandas as pd
#import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np
import settings
import ast
import scipy as sp
import querying.feature_vector as fv
import os

def string_to_floats(string_feature_vectors):
    float_feature_vectors = []
    for i in range(len(string_feature_vectors)):
        float_vec = ast.literal_eval(string_feature_vectors[i])
        flat_list = [item for sublist in float_vec for item in (sublist if isinstance(sublist, list) else [sublist])]
        float_feature_vectors.append(flat_list)
    return float_feature_vectors


def read_feature_vector():
    csv_df = pd.read_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH)
    all_shapeclasses = csv_df.get(key='shapeclass')
    all_shapeclasses = list(all_shapeclasses)
    all_filenames = csv_df.get(key='filename')
    all_filenames = list(all_filenames)
    string_feature_vectors = csv_df.get(key="featurevector")
    string_feature_vectors = string_feature_vectors.tolist()
    float_feature_vectors = string_to_floats(string_feature_vectors)
    return float_feature_vectors, all_shapeclasses, all_filenames

def plot_fvs(reduced_feature_vectors, shape_classes, plot = False):
    x = [a[0] for a in reduced_feature_vectors]
    y = [a[1] for a in reduced_feature_vectors]

    #inertias = []
   # kmeans = KMeans(n_clusters=59, n_init='auto')
    #kmeans.fit(reduced_feature_vectors)
    #inertias.append(kmeans.inertia_)

    colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#ff9896", "#aec7e8", "#ffbb78", "#98df8a", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7f9cf4",
    "#ab63f9", "#4ec2cd", "#d7263d", "#509e2f", "#be4197",
    "#2a2a72", "#8c8c3b", "#4a4a30", "#7a6a53", "#462748",
    "#6b82a6", "#455d69", "#bbd3e4", "#ff6633", "#ffae6a",
    "#ffa07a", "#8a2be2", "#a020f0", "#6a5acd", "#483d8b",
    "#dda0dd", "#ff1493", "#db7093", "#ff4500", "#ff8c00",
    "#ffd700", "#adff2f", "#32cd32", "#006400", "#008000",
    "#4b0082", "#8a2be2", "#ff00ff", "#8b4513", "#a0522d",
    "#d2691e", "#cd853f", "#d2b48c", "#ffebcd", "#e9967a",
    "#b22222", "#dc143c", "#ff7f50", "#fa8072", "#ffc0cb",
    "#fff5e1", "#ffffe0", "#f0e68c", "#e6e6fa", "#fffafa"
    ]
    classes_set = sorted(set(shape_classes))
    print(len(classes_set))
    print(len(colors))
    color_palette = dict()
    for i in range(len(classes_set)):
        color_palette[classes_set[i]] = colors[i]
    
    plot_colors = []
    for el in shape_classes:
        plot_colors.append(color_palette[el])
    # for el in kmeans.labels_:
    #     plot_colors.append(color_palette[el])

    if plot:
        data = {
            'X': x,
            'Y': y,
            'Label': shape_classes
        }
        fig = px.scatter(data, x='X', y='Y', color='Label', color_discrete_sequence=colors, hover_name=shape_classes)
        fig.show()

    return plot_colors, x, y, shape_classes


def apply_tsne(plot = False):
    #Read feature vectors from CSV, convert from string to floats and flatten the array
    feature_vectors, shape_classes, file_names = read_feature_vector()
    feature_vectors = np.array(feature_vectors)
    #Reduce dimensions to 2, using tsne
    reduced_feature_vectors = tsne.tsne(feature_vectors, 2, initial_dims=len(feature_vectors[0]), perplexity=25.0)
    #Plot the reduces feature vectors
    plot_colors, x, y, shape_classes = plot_fvs(reduced_feature_vectors, shape_classes, plot=plot)

    return plot_colors, x, y, shape_classes, file_names

def apply_knn(query_shape_path, is_analyze = False):
    print(query_shape_path)
    csv_df = pd.read_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH)
    if is_analyze:
        #query_feature_vector = pd.read_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH, skiprows = lambda x: x not in [query_shape_path])
        query_data = csv_df[csv_df['filename']==query_shape_path]
        query_feature_vector = query_data.get(key='featurevector')
        query_feature_vector = query_feature_vector.tolist()
        query_feature_vector = ast.literal_eval(query_feature_vector[0])
    else:
        query_feature_vector = fv.generate_query_feature_vector(query_shape_path)
    
    query_feature_vector = [item for sublist in query_feature_vector for item in (sublist if isinstance(sublist, list) else [sublist])]

    feature_vectors, shape_classes, file_names = read_feature_vector()

    kdtree = sp.spatial.KDTree(feature_vectors)
    print(query_feature_vector)
    knn_distances, knn_indices = kdtree.query(query_feature_vector, k=settings.k)

    k_nearest_shapes = []
    k_nearest_classes = []
    for index in knn_indices:
        k_nearest_shapes.append(file_names[index])
        k_nearest_classes.append(os.path.basename(os.path.dirname(file_names[index])))
        

    return k_nearest_shapes, knn_distances, k_nearest_classes
