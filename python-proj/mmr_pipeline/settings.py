import numpy as np
### GLOBAL SETTINGS FILE ###

# Data I/O settings
DB_ORIGINAL_DIRECTORY           = R'data/original/'
DB_NORMALIZED_DIRECTORY         = R'data/normalized/'
ANALYZE_RESULTS_FOLDER          = R'analyze-results/'
CSV_UNNORMALIZED_OUTPUT_PATH    = R'analyze-results/unnormalized_data.csv'
CSV_NORMALIZED_OUTPUT_PATH      = R'analyze-results/normalized_data.csv'
CSV_OUTLIERS_OUTPUT_PATH        = R'analyze-results/outliers.csv'
CSV_AVERAGE_SHAPE_OUTPUT_PATH   = R'analyze-results/averages.csv'
CSV_SHAPE_DESCRIPTORS_OUTPUT_PATH   = R'analyze-results/shape_property_descriptors.csv'
CSV_ELEMENTARY_DESCRIPTORS_OUTPUT_PATH   = R'analyze-results/elementary_descriptors.csv'
CSV_FEATURE_VECTORS_OUTPUT_PATH   = R'analyze-results/feature_vectors.csv'
CSV_STANDARDIZATION_PATH   = R'analyze-results/standardization_data.csv'
CSV_METRICS_CLASSES_PATH   = R'analyze-results/metrics/'
CSV_METRICS_KNN_PATH   = R'analyze-results/metrics_knn/'
CSV_QUERY_RESULTS_PATH   = R'analyze-results/query_results.csv'
CSV_QUERY_RESULTS_KNN_PATH   = R'analyze-results/query_results_knn.csv'
HISTOGRAM_OUTPUT_PATH           = R'analyze-results/histograms/'
HISTOGRAM_OUTPUT_PATH_DESCRIPTOR           = R'analyze-results/histograms/descriptors'

# Mesh refinement settings
MESH_VERTEX_COUNT   = 5000
MESH_VERTEX_EPSILON = 200

#Shape property descriptor settings
SPD_BIN_COUNT = 30
SPD_NORMALIZE_VALUES = [360.0, np.sqrt(3.0) / 2.0, np.sqrt(3.0), np.sqrt(np.sqrt(3.0) / 2.0), np.cbrt(((1.0 / 3.0) * 0.5 * 1.0))]
SPD_CLASSES = ["A3", "D1", "D2", "D3", "D4"]

#Querying settings
k = 6 #querying size

#Analyze results settings
ANALYZE_METRICS = ["TP", "FP", "FN"]
CLASS_SIZE_LIMIT = 0