import csv_manager, settings

def generate_csv(regular: bool, normalized: bool) -> None:
    if regular:
        print("GENERATING CSV FILE!")
        csv_manager.reset_csv_file(settings.CSV_UNNORMALIZED_OUTPUT_PATH)
        csv_manager.save_all_to_csv(settings.CSV_UNNORMALIZED_OUTPUT_PATH, False)
    
    if normalized:
        print("GENERATING NORMALIZED CSV FILE!")
        #csv_manager.reset_csv_file(settings.CSV_NORMALIZED_OUTPUT_PATH)
        csv_manager.save_all_to_csv(settings.CSV_NORMALIZED_OUTPUT_PATH, True)

def generate_csv_descriptors(elementary: bool, property: bool) -> None:
    if property:
        print("GENERATING DESCRIPTOR CSV FILE!")
        csv_manager.reset_csv_file(settings.CSV_SHAPE_DESCRIPTORS_OUTPUT_PATH)
        csv_manager.save_descriptors_to_csv(settings.CSV_SHAPE_DESCRIPTORS_OUTPUT_PATH)
    if elementary:
        csv_manager.reset_csv_file(settings.CSV_ELEMENTARY_DESCRIPTORS_OUTPUT_PATH)
        csv_manager.save_elementary_descriptors_to_csv(settings.CSV_ELEMENTARY_DESCRIPTORS_OUTPUT_PATH)