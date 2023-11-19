import settings
import glob, os
import querying.feature_distance as fd
import csv_manager
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import ast
import plotly.express as px
from sklearn.metrics import roc_curve, auc
import data_generation.generate_histograms as generate_histograms
import matplotlib.pyplot as plt

def get_database_size(database_path):
    folders = glob.glob(f'{database_path}/*')
    folders = sorted(folders)

    #Get database size
    database_size = 0
    for folder in folders:      
        if len(glob.glob(f'{folder}/*.obj')) < settings.CLASS_SIZE_LIMIT: continue
        database_size += len(glob.glob(f'{folder}/*.obj'))
    
    return database_size


def update_metrics(all_metrics, classname, TP, FP, database_size, query_size):
    all_metrics[classname]["TP"] += TP
    all_metrics["total"]["TP"] += TP
    all_metrics[classname]["FP"] += FP
    all_metrics["total"]["FP"] += FP
    all_metrics[classname]["FN"] += query_size - TP
    all_metrics["total"]["FN"] += query_size - TP
    all_metrics[classname]["TN"] += database_size - query_size- FP
    all_metrics["total"]["TN"] += database_size - query_size - FP

def read_query_result_csv(query_shape_path, k, query_result_path) -> list[str]:
    csv_df = pd.read_csv(query_result_path)
    #print(query_shape_path)
    query_data = csv_df[csv_df['querypath']==query_shape_path]
    query_results = query_data.get(key='queryresults')
    query_results = query_results.tolist()
    query_results = ast.literal_eval(query_results[0])
   # print(query_results[0:k])
    return query_results[0:k]

def get_all_query_results(k_max, knn = False):
    for k in range(1, k_max + 1):
        get_query_results(k, knn)

def get_query_results(k, knn = False):
    metrics_path = settings.CSV_METRICS_CLASSES_PATH
    if knn: metrics_path = settings.CSV_METRICS_KNN_PATH
    #Prepare csv file

    csv_manager.reset_csv_file(metrics_path + "k" + str(k) + ".csv")
    headers = ["shapeclass", "TP", "FP", "FN", "TN", "filecount"]
    csv_manager.save_to_csv(metrics_path + "k" + str(k) + ".csv", headers)

    #Get reference to folders containing the data
    folders = glob.glob(f'{settings.DB_NORMALIZED_DIRECTORY}/*')
    folders = sorted(folders)

    #Prepare dictionary to save data
    all_metrics = dict(dict())
    initiate_dictionary("total", all_metrics)

    #Get database size
    database_size = get_database_size(settings.DB_NORMALIZED_DIRECTORY)
    
    for folder in folders:     
        classname = os.path.basename(folder)
        files = glob.glob(f'{folder}/*.obj')
        file_count = len(files)
        if file_count < settings.CLASS_SIZE_LIMIT: continue
        initiate_dictionary(classname, all_metrics)
        all_metrics[classname]["filecount"] = file_count
        all_metrics["total"]["filecount"] = database_size

        for file in files:      
            #k_nearest_shapes, k_nearest_distances, k_nearest_shape_classes = fd.read_feature_vector(file, True)
            query_result_path = settings.CSV_QUERY_RESULTS_PATH
            if knn: query_result_path = settings.CSV_QUERY_RESULTS_KNN_PATH
            k_nearest_shape_classes = read_query_result_csv(file, k, query_result_path)
            TP, FP = 0, 0
            for query_result in k_nearest_shape_classes:
                if (query_result == classname):
                    TP += 1
                else:
                    FP += 1
            update_metrics(all_metrics, classname, TP, FP, database_size, k)
            
        save_metrics_to_csv(all_metrics, classname, metrics_path + "k" + str(k) + ".csv")

    save_metrics_to_csv(all_metrics, "total", metrics_path + "k" + str(k) + ".csv")

def initiate_dictionary(classname, dictionary):
        dictionary[classname] = {}
        dictionary[classname]["TP"] = 0
        dictionary[classname]["FP"] = 0
        dictionary[classname]["FN"] = 0
        dictionary[classname]["TN"] = 0
        dictionary[classname]["filecount"] = 0

def save_metrics_to_csv(metrics, shape_class, csv_filepath):
    TP = metrics[shape_class]["TP"]
    FP = metrics[shape_class]["FP"]
    FN = metrics[shape_class]["FN"]
    TN = metrics[shape_class]["TN"]
    filecount = metrics[shape_class]["filecount"]
    data =[shape_class, TP, FP, FN, TN, filecount]
    csv_manager.save_to_csv(csv_filepath, data)

def compute_metrics(TP, FP, FN, TN, filecount, precisions, recalls, accuracies, specificities, sensitivities):
    database_size = get_database_size(settings.DB_NORMALIZED_DIRECTORY)
    precisions.append(TP/(TP+FP)) # How many of the class were actually the class
    recalls.append(TP/(TP+FN)) # How many of the classs were guessed to be the class
    accuracies.append((TP+TN)/database_size)
    specificities.append(TN/(FP+TN)) # How many of the 'False' classes are correctly classified as being False
        #E.g. how many healthy people are identified as not being sick
    sensitivities.append(TP/(TP+FN)) # How many of the 'True' classes are actually returned as True
        #E.g. how many sick people are identified as being sick

    return

def normalize_values(TPs, FPs, FNs, TNs, filecounts, shape_classes):
    for i in range(len(shape_classes)):
        TPs[i] = TPs[i] / filecounts[i]
        FPs[i] = FPs[i] / filecounts[i]
        FNs[i] = FNs[i] / filecounts[i]
        TNs[i] = TNs[i] / filecounts[i]

def round_values(TPs, FPs, FNs, TNs, shape_classes, precisions, recalls, accuracies, specificities, sensitivities):
    for i in range(len(shape_classes)):
        TPs[i] = round(TPs[i], 2)
        FPs[i] = round(FPs[i], 2)
        FNs[i] = round(FNs[i], 2)
        TNs[i] = round(TNs[i], 2)
        precisions[i] = round(precisions[i], 3)
        recalls[i] = round(recalls[i], 3)
        accuracies[i] = round(accuracies[i], 4)
        specificities[i] = round(specificities[i], 6)
        sensitivities[i] = round(sensitivities[i], 6)

def save_final_metrics(k_max, knn= False):
    csv_path = settings.CSV_METRICS_CLASSES_PATH
    if knn: csv_path = settings.CSV_METRICS_KNN_PATH
    for k in range(1, k_max + 1):
        save_all_metrics(csv_path + "k" + str(k) + ".csv", csv_path + "met" + str(k) + ".csv")

def save_all_metrics(query_data_path, metric_output_path,):
    csv_manager.reset_csv_file(metric_output_path)
    headers = ["shapeclass", "precision", "recall", "accuracy", "specificity", "sensitivity"]
    csv_manager.save_to_csv(metric_output_path, headers)

    csv_df = pd.read_csv(query_data_path)
    shape_classes = csv_df.get(key='shapeclass')
    shape_classes = list(shape_classes)
    TPs = csv_df.get(key="TP")
    TPs = list(TPs)
    FPs = csv_df.get(key="FP")
    FPs = list(FPs)
    FNs = csv_df.get(key="FN")
    FNs = list(FNs)
    TNs = csv_df.get(key="TN")
    TNs = list(TNs)
    filecounts = csv_df.get(key="filecount")
    filecounts = list(filecounts)

        #normalize data
    normalize_values(TPs, FPs, FNs, TNs, filecounts, shape_classes)

    precisions, recalls, accuracies, specificities, sensitivities = [], [], [], [], []
    for i in range(len(shape_classes)):
        compute_metrics(TPs[i], FPs[i], FNs[i], TNs[i], filecounts[i], precisions, recalls, accuracies, specificities, sensitivities)       
    round_values(TPs, FPs, FNs, TNs, shape_classes, precisions, recalls, accuracies, specificities, sensitivities)
    
    for i in range(len(shape_classes)):
        data =[shape_classes[i], precisions[i], recalls[i], accuracies[i], specificities[i], sensitivities[i]]
        csv_manager.save_to_csv(metric_output_path, data)


def plot_ROC_curve(max_k, knn=False):
    csv_path = settings.CSV_METRICS_CLASSES_PATH
    if knn: csv_path = settings.CSV_METRICS_KNN_PATH
    specificities = dict()
    sensitivities = dict()
    for k in range(1, max_k + 1):
        metrics_path = csv_path + "met" + str(k) + ".csv"
        csv_df = pd.read_csv(metrics_path)
        shape_classes = csv_df.get(key="shapeclass")
        shape_classes = list(shape_classes)
        csv_specificities = csv_df.get(key="specificity")
        csv_specificities = list(csv_specificities)
        csv_sensitivities = csv_df.get(key="sensitivity")
        csv_sensitivities = list(csv_sensitivities)

        for i in range(len(shape_classes)):
            if(k == 1):
                specificities[shape_classes[i]] = []
                sensitivities[shape_classes[i]] = []
            specificities[shape_classes[i]].append(csv_specificities[i])
            sensitivities[shape_classes[i]].append(csv_sensitivities[i])

    #spec = specificities['Cup']
    #sens = sensitivities['Cup']

    aucs = dict()
    spec = dict()
    sens = dict()
    for shape in shape_classes:
        #print(shape)
        specificity = specificities[shape]
        sensitivity = sensitivities[shape]

        fit_specificity = [(el - np.min(specificity)) / (np.max(specificity) - np.min(specificity)) for el in specificity]
        fit_sensitivity = [(el - np.min(sensitivity)) / (np.max(sensitivity) - np.min(sensitivity)) for el in sensitivity]

        spec[shape] = fit_specificity
        sens[shape] = fit_sensitivity

        a = auc(fit_specificity, fit_sensitivity)
        aucs[shape] = a

    sorted_aucs = dict(sorted(aucs.items(), key=lambda item: item[1]))
    sorted_aucs_keys = list(sorted_aucs.keys())

    print(sorted_aucs_keys[0])
    ind=sorted_aucs_keys[0]
    ind = 'total'

    fig = px.area(
    x=spec[ind], y=sens[ind],
    title=f'ROC Curve (AUC={auc(spec[ind], sens[ind]):.4f})',
    labels=dict(x='specificity', y='sensitivity'),
    width=700, height=500, markers=True
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


def create_metric_table(knn = False):
    #TP,FP,FN,TN,filecount
    if knn: csv_df = pd.read_csv(settings.CSV_METRICS_KNN_PATH + "/k5.csv")
    else:
        csv_df = pd.read_csv(settings.CSV_METRICS_CLASSES_PATH + "/k5.csv")

    shape_classes = csv_df.get(key='shapeclass')
    shape_classes = list(shape_classes)
    TPs = csv_df.get(key="TP")
    TPs = list(TPs)
    FPs = csv_df.get(key="FP")
    FPs = list(FPs)
    FNs = csv_df.get(key="FN")
    FNs = list(FNs)
    TNs = csv_df.get(key="TN")
    TNs = list(TNs)
    filecounts = csv_df.get(key="filecount")
    filecounts = list(filecounts)

    #normalize data
    normalize_values(TPs, FPs, FNs, TNs, filecounts, shape_classes)

    precisions, recalls, accuracies, specificities, sensitivities = [], [], [], [], []
    for i in range(len(shape_classes)):
        compute_metrics(TPs[i], FPs[i], FNs[i], TNs[i], filecounts[i], precisions, recalls, accuracies, specificities, sensitivities)       
    round_values(TPs, FPs, FNs, TNs, shape_classes, precisions, recalls, accuracies, specificities, sensitivities)

    fig = go.Figure(data=[go.Table(header=dict(values=['class name', 'file count', 'TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'accuracy', 'specificity', 'sensitivity'],
                                               fill_color='#fa9e61'),
                 cells=dict(values=[shape_classes, filecounts, TPs, FPs, FNs, TNs, precisions, recalls, accuracies, specificities, sensitivities],
                            fill=dict(color=['#ffccab', '#fff4ed']),
                            ))])
    fig.show()



def create_precision_histogram():
    csv_df = pd.read_csv(settings.CSV_METRICS_CLASSES_PATH + "/met5.csv")
    precisions = csv_df.get(key="precision")
    precisions = list(precisions)
    precisions = np.array(precisions)
    
    print(precisions)
    shape_classes = csv_df.get(key='shapeclass')
    shape_classes = list(shape_classes)
    shape_classes = np.array(shape_classes)
    print(shape_classes)

    c = np.rec.fromarrays([precisions, shape_classes])
    c.sort()

    print(c.f0[0])
    print(c.f0[len(c.f0) - 1])

    f0 = c.f0
    f1 = c.f1

    for i in range(len(f1)):
        if c.f1[i] == "total":
            f1 = np.delete(f1, i)
            precision = f0[i]
            f0 = np.delete(f0, i)
            f0 = np.insert(f0, 0, precision)
            f1 = np.insert(f1, 0, "total")
            print("sloep")
            break

    plt.bar(f1, f0, color = "#db6b6b") # type: ignore

    plt.xticks(range(len(f1)), f1, size=6.0, rotation=90) # type: ignore

    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f'{settings.HISTOGRAM_OUTPUT_PATH}{"precision_classes.png"}')
    print(f'{settings.HISTOGRAM_OUTPUT_PATH}{"precision_classes.png"}')
    #plt.clf()

    plt.show()

    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.title(title)

    # bar =plt.bar(range(len(value_df)), value_df.values, align='center', color = "#db6b6b") # type: ignore

    # plt.xticks(range(len(value_df)), value_df.index.values, size=6.0, rotation=90) # type: ignore

    # plt.autoscale()
    # plt.tight_layout()
    # plt.savefig(f'{settings.HISTOGRAM_OUTPUT_PATH}/{"precision_classes.png"}')
    # plt.clf()

