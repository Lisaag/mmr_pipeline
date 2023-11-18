import argparse
import mesh_loader
from data_generation import generate_csv, generate_histograms, calculate_averages
import settings
from querying import feature_vector, feature_distance
from scalability import clustering
from evaluation import quality_analyzer, query_result_saver
from feature_extraction import test_shapes
import faulthandler

PARSER = argparse.ArgumentParser(prog='MMRPipeline')
PARSER.add_argument('--display', action='store', choices=['all', 'outliers', 'averages', 'file'])
PARSER.add_argument('--file', action='store')
PARSER.add_argument('--analyze', action='store', choices=['get_data', 'generate_table', 'generate_query_results', 'generate_metrics', 'generate_roc', 'descriptors'])
PARSER.add_argument('--generate-csv', action='store', choices=['all', 'norm', 'outliers', 'el_descriptors', 'prop_descriptors', 'feature_vectors'])
PARSER.add_argument('--generate-figures', action='store_true')
PARSER.add_argument('--normalize', action='store_true')
PARSER.add_argument('--query', action='store', choices=['euclidean_distance'])
PARSER.add_argument('--cluster', action='store', choices=['tsne'])


if __name__ == '__main__':
    args = PARSER.parse_args() # type: ignore

    print(f'Normalization { "enabled" if args.normalize else "disabled" }')

    if (args.display):
        if (args.display == 'all'):
            mesh_loader.display_all(args.normalize)
        if (args.display == 'outliers'):
            mesh_loader.display_csv(settings.CSV_OUTLIERS_OUTPUT_PATH, args.normalize)
        if (args.display == 'averages'):
            mesh_loader.display_csv(settings.CSV_AVERAGE_SHAPE_OUTPUT_PATH, args.normalize)
        if (args.display == 'file'):
            print(f'Displaying object file {args.file}')
            mesh_loader.display("data/normalized/Bicycle/D00077.obj", args.normalize)
            exit()

    if (args.generate_csv):
        if (args.generate_csv == 'norm'):
            faulthandler.enable()
            generate_csv.generate_csv(False, True)
        if (args.generate_csv == 'outliers'):
            calculate_averages.calculate_average_shape()
        if (args.generate_csv == 'all'):
            generate_csv.generate_csv(True, False)
            calculate_averages.calculate_average_shape()
        if (args.generate_csv == 'el_descriptors'):
            generate_csv.generate_csv_descriptors(True, False)
        if (args.generate_csv == 'prop_descriptors'):
            generate_csv.generate_csv_descriptors(False, True)
        if (args.generate_csv == 'feature_vectors'):
            feature_vector.write_standardized_descriptors_to_csv()

    if (args.generate_figures):
        generate_histograms.generate_histograms()

    if(args.query):
        if(args.query == 'euclidean_distance'):
            feature_distance.read_feature_vector("data/normalized/Chess/D01053.obj")

    # if(args.cluster):
    #     if(args.cluster == 'tsne'):
    #         clustering.apply_tsne()

    if(args.analyze):
        if(args.analyze == 'get_data'):
            quality_analyzer.get_all_query_results(10, knn=False)
        elif(args.analyze == 'generate_table'):
            quality_analyzer.create_metric_table(knn = False)
        elif(args.analyze == 'generate_query_results'):
            query_result_saver.save_all_query_results(knn=False)
        elif(args.analyze == 'generate_metrics'):
            quality_analyzer.save_final_metrics(10, knn=False)
        elif(args.analyze == 'generate_roc'):
            quality_analyzer.plot_ROC_curve(2, knn=False)
        elif(args.analyze == 'descriptors'):
            test_shapes.get_descriptors_primary_shape()

