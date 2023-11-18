import settings
import glob, os
import querying.feature_distance as fd
import csv_manager
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import scalability.clustering as clustering

def save_all_query_results(knn=False):
    #Prepare csv file
    if knn:
        csv_manager.reset_csv_file(settings.CSV_QUERY_RESULTS_KNN_PATH)
        headers = ["querypath", "queryclass", "queryresults"]
        csv_manager.save_to_csv(settings.CSV_QUERY_RESULTS_KNN_PATH, headers)
    else:
        csv_manager.reset_csv_file(settings.CSV_QUERY_RESULTS_PATH)
        headers = ["querypath", "queryclass", "queryresults"]
        csv_manager.save_to_csv(settings.CSV_QUERY_RESULTS_PATH, headers)

    #Get reference to folders containing the data
    folders = glob.glob(f'{settings.DB_NORMALIZED_DIRECTORY}/*')
    folders = sorted(folders)
    
    for folder in folders:     
        classname = os.path.basename(folder)
        files = glob.glob(f'{folder}/*.obj')
        for file in files:
            if knn:
                k_nearest_shapes, k_nearest_distances, shape_classes = clustering.apply_knn(file, True)
                data = [file, classname, shape_classes]
                csv_manager.save_to_csv(settings.CSV_QUERY_RESULTS_KNN_PATH, data)
            else:
                k_nearest_shapes, k_nearest_distances, k_nearest_shape_classes = fd.read_feature_vector(file, True)
            
                data = [file, classname, k_nearest_shape_classes]
                csv_manager.save_to_csv(settings.CSV_QUERY_RESULTS_PATH, data)