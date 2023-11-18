import ast
import pandas as pd
import numpy as np
import querying.feature_vector as fv
import settings
import os

def isIterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False

#Calculate Euclidean distance between two feature vectors
def compute_distance(query_vector, compare_vector, all_distances):
    compare_vector = ast.literal_eval(compare_vector)
    for i in range(len(query_vector)):
        #if i is an iterable, this is a shape property descriptor, bc this is saved in an array  --> use Earth Mover's Distance
        if isIterable(query_vector[i]):
            all_distances[i].append(compute_emd(query_vector[i], compare_vector[i]))
        #else i is a elementary descriptur, thus a single floating point value --> use Euclidean distance
        else:
            all_distances[i].append(np.sqrt(np.power(query_vector[i] - compare_vector[i], 2)))


def compute_emd(features_1, features_2):
    i, j = 0, 1
    flow = [[0 for _ in range(len(features_1))] for _ in range(len(features_1))]
    difference = [0] * len(features_1)
    row = [0] * len(features_1)

    # Initialize empty flow matrix
    for p in range(len(features_1)):
        flow[p] = row.copy()
        flow[p][p] = min(features_1[p], features_2[p])
        difference[p] = features_1[p] - features_2[p]

    # Fill out the flow matrix by spreading differences
    while i + j < 2 * (len(features_1) - 1):
        if difference[i] > 0 and difference[j] < 0:
            if difference[i] <= -difference[j]:
                flow[j][i] = difference[i]
                difference[j] += difference[i]
                difference[i] = 0
                i += 1
                j = i + 1
            else:
                flow[j][i] = -difference[j]
                difference[i] += difference[j]
                difference[j] = 0
                if j < (len(features_1) - 1):
                    j += 1
                else:
                    i += 1
                    j = i + 1
        elif difference[i] < 0 and difference[j] > 0:
            if -difference[i] < difference[j]:
                flow[j][i] = -difference[i]
                difference[j] += difference[i]
                difference[i] = 0
                i += 1
                j = i + 1
            else:
                flow[i][j] = difference[j]
                difference[i] += difference[j]
                difference[j] = 0
                if j < len(features_1) - 1:
                    j += 1
                else:
                    i += 1
                    j = i + 1
        elif difference[i] == difference[j]:
            i += 1
            j = i + 1
        else:
            if j < len(features_1) - 1:
                j += 1
            else:
                i += 1
                j = i + 1

    # Compute sum of distance times flow
    work = 0
    for p in range(len(features_1)):
        for q in range(len(features_1)):
            work += abs(p - q) * flow[p][q]
    
    # 'Normalize' by dividing by total flow
    total_flow = 0
    for i in range(len(features_1)):
        for j in range(len(features_1)):
            total_flow += flow[i][j]
    emd = work / total_flow

    return emd

def weigh_distances(all_distances, all_filenames):
    all_means = []
    all_sd = [] #all standard deviations
    all_total_distances = dict()
    for i in range(len(all_distances)):      
        all_means.append(fv.compute_mean(all_distances[i]))
        all_sd.append(fv.compute_standard_deviation(all_distances[i], all_means[i]))
    
    for j in range(len(all_distances[0])):
        total_distance = 0
        for i in range(len(all_distances)):    
            total_distance += (all_distances[i][j] - all_means[i]) / all_sd[i]
        all_total_distances[all_filenames[j]] = total_distance

    return all_total_distances    



def read_feature_vector(query_shape_path, is_analyze = False):
    csv_df = pd.read_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH)
    if is_analyze:
        #query_feature_vector = pd.read_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH, skiprows = lambda x: x not in [query_shape_path])
        query_data = csv_df[csv_df['filename']==query_shape_path]
        query_feature_vector = query_data.get(key='featurevector')
        query_feature_vector = query_feature_vector.tolist()
        query_feature_vector = ast.literal_eval(query_feature_vector[0])
    else:
        query_feature_vector = fv.generate_query_feature_vector(query_shape_path)
    
    all_filenames = csv_df.get(key='filename')
    all_filenames = list(all_filenames)
    shape_classes = csv_df.get(key='shapeclass')
    shape_classes = list(shape_classes)
    all_feature_vectors = csv_df.get(key="featurevector")
    #all_feature_vectors = all_feature_vectors.tolist()
    #query_shape_index = all_filenames.index(query_shape_path)

    all_distances = []
    for i in range(len(query_feature_vector)): all_distances.append([])

    for i in range(len(all_filenames)):
        compute_distance(query_feature_vector, all_feature_vectors[i], all_distances)

    distances = weigh_distances(all_distances, all_filenames)
    
    distances = sorted(distances.items(), key=lambda item: item[1])
    k_nearest_shapes = []
    k_nearest_distances = []
    k_nearest_classes = []
    for k in range(settings.k):
        print(distances[k][0])
        k_nearest_shapes.append(distances[k][0])
        k_nearest_distances.append(distances[k][1])
        k_nearest_classes.append(os.path.basename(os.path.dirname(distances[k][0])))

    return k_nearest_shapes, k_nearest_distances, k_nearest_classes





