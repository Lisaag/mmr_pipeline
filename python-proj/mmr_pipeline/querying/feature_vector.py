import pandas as pd
import settings
import numpy as np
import csv_manager
import vedo
import data_generation.generate_histograms as generate_histograms
import feature_extraction.elementary_descriptors as elementary_descriptors
import feature_extraction.shape_property_descriptors as shape_property_descriptors
import settings

def compute_mean(descriptors) -> float:
    sum = np.sum(descriptors)
    length = len(descriptors)
    return sum / length

def compute_standard_deviation(descriptors, mean):
    mean_subtr_value = 0
    for d in descriptors:
        mean_subtr_value += np.power(d - mean, 2)
    standard_deviation = np.sqrt(mean_subtr_value/len(descriptors))
    return standard_deviation

def read_elementary_descriptors_csv():
    csv_df = pd.read_csv(settings.CSV_ELEMENTARY_DESCRIPTORS_OUTPUT_PATH)
    csv_keys = csv_df.keys()[2:] #trim first element, bc this represents the shape class and is not relevant
    all_descriptors = []

    for key in csv_keys:
        descriptors = csv_df.get(key=key)
        all_descriptors.append(descriptors)

    return all_descriptors

def standardize_elementary_descriptors():
    all_descriptors = read_elementary_descriptors_csv()
    ind = 0

    #prepare csv with standardization values per descriptor
    csv_manager.reset_csv_file(settings.CSV_STANDARDIZATION_PATH)
    csv_manager.save_to_csv(settings.CSV_STANDARDIZATION_PATH, ["mean", "standard_deviation", "min", "max"])

    #loop over descriptor of all objects
    for descriptors in all_descriptors:
        ind+=1
        mean = compute_mean(descriptors)
        standard_deviation = compute_standard_deviation(descriptors, mean)
        
        ##Apply standardization
        for i in range(len(descriptors)):
            descriptors[i] = (descriptors[i] - mean) / standard_deviation
        
        min = np.min(descriptors)
        max = np.max(descriptors)
        csv_manager.save_to_csv(settings.CSV_STANDARDIZATION_PATH, [mean, standard_deviation, min, max])
        #apply min-max normalization to get descriptor within a 0-1 range
        for i in range(len(descriptors)):
            norm_desc = (descriptors[i] - min) / (max - min)
            descriptors[i] = norm_desc

    return all_descriptors

def normalize_shape_propery_descriptors():
    descriptor_classes = settings.SPD_CLASSES
    normalize_length = settings.SPD_NORMALIZE_VALUES
    all_bins = []
    for i in range(len(descriptor_classes)):
        bins = generate_histograms.generate_all_descriptor_histograms(descriptor_classes[i], normalize_length[i])
        all_bins.append(bins)

    return all_bins #this array is of shape [len(descriptor_classes), len(shapes), len(bins)]
        

def write_standardized_descriptors_to_csv():
    csv_manager.reset_csv_file(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH)

    csv_df = pd.read_csv(settings.CSV_ELEMENTARY_DESCRIPTORS_OUTPUT_PATH)
    headers = ["shapeclass", "filename", "featurevector"]
    csv_manager.save_to_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH, headers)

    all_class_names = csv_df.get(key='shapeclass')
    all_file_names = csv_df.get(key='filename')
    all_elementary_descriptors = standardize_elementary_descriptors()
    all_shape_property_descriptors = normalize_shape_propery_descriptors()

    #loop through each shape in the normalized database
    for i in range(len(all_class_names)): #type: ignore
        feature_vector = []
        for descriptor in all_elementary_descriptors:
            feature_vector.append(descriptor[i])
        for descriptor in all_shape_property_descriptors:
            feature_vector.append(descriptor[i])
        
        csv_data = [all_class_names[i], all_file_names[i], feature_vector] #type: ignore
        csv_manager.save_to_csv(settings.CSV_FEATURE_VECTORS_OUTPUT_PATH, csv_data)


def generate_query_feature_vector(query_shape_path):
    query_object = vedo.Mesh(query_shape_path)
    feature_vector = []
    el_descriptors = [] #rectangularity,compactness,convexity,eccentricity
    el_descriptors.append(elementary_descriptors.calculate_rectangularity(query_object))
    el_descriptors.append(elementary_descriptors.calculate_compactness(query_object))
    el_descriptors.append(elementary_descriptors.calculate_convexity(query_object))
    el_descriptors.append(elementary_descriptors.calculate_eccentricity(query_object))

    csv_df = pd.read_csv(settings.CSV_STANDARDIZATION_PATH)
    csv_mean = csv_df.get(key='mean')
    csv_sd = csv_df.get(key='standard_deviation')
    csv_min = csv_df.get(key='min')
    csv_max = csv_df.get(key='max')

    for i in range(len(el_descriptors)):
        el_descriptors[i] = (el_descriptors[i] - csv_mean[i]) / csv_sd[i] #type: ignore
        el_descriptors[i] = (el_descriptors[i] - csv_min[i]) / (csv_max[i] - csv_min[i]) #type: ignore
        feature_vector.append(el_descriptors[i])
    

    sp_descriptors = []
    sp_descriptors.append(shape_property_descriptors.shape_property_a3(query_object))
    sp_descriptors.append(shape_property_descriptors.shape_property_d1(query_object))
    sp_descriptors.append(shape_property_descriptors.shape_property_d2(query_object))
    sp_descriptors.append(shape_property_descriptors.shape_property_d3(query_object))
    sp_descriptors.append(shape_property_descriptors.shape_property_d4(query_object))
    
    normalized_sp_descriptors = []
    for i in range(len(sp_descriptors)):
        generate_histograms.generate_descriptor_histogram([sp_descriptors[i]], settings.SPD_NORMALIZE_VALUES[i], normalized_sp_descriptors)
        feature_vector.append(normalized_sp_descriptors[i])

    return feature_vector