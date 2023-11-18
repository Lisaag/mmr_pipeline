import csv
import re
import numpy as np
import helpers, settings, mesh_loader
from normalization import rotation, scaling
from feature_extraction import shape_property_descriptors as spd
from feature_extraction import elementary_descriptors as ed


def reset_csv_file(filename: str) -> None:
    helpers.create_output_folder(settings.ANALYZE_RESULTS_FOLDER)
    with open(filename, 'w', newline='') as csvfile:
        csvfile.truncate()

def save_to_csv(filename: str, data: list) -> None:
    helpers.create_output_folder(settings.ANALYZE_RESULTS_FOLDER)
    with open(filename, 'a', newline='') as csvfile:
        datawriter = csv.writer(
            csvfile,
            delimiter=',',
            quotechar='\"',
            quoting=csv.QUOTE_MINIMAL
        )

        datawriter.writerow(data)

def save_descriptors_to_csv(filename: str):
    headers = ["shapeclass", "filename", "A3", "D1", "D2", "D3", "D4"]
    FILES = helpers.list_db_files(settings.DB_NORMALIZED_DIRECTORY)
    save_to_csv(filename, headers)

    for file in FILES:
        mesh = mesh_loader.load_mesh(file)
        #print(len(mesh.points()))
        splits = re.split('[\\\\/]', file)
        foldername = splits[-2]
        A3 = spd.shape_property_a3(mesh)
        D1 = spd.shape_property_d1(mesh)
        D2 = spd.shape_property_d2(mesh)
        D3 = spd.shape_property_d3(mesh)
        D4 = spd.shape_property_d4(mesh)

        csv_data = [ 
            foldername,
            file,
            A3,
            D1,
            D2,
            D3,
            D4
        ]
            
        save_to_csv(filename, csv_data)

def save_elementary_descriptors_to_csv(filename: str):
    headers = ["shapeclass", "filename", "rectangularity", "compactness", "convexity", "eccentricity"]
    FILES = helpers.list_db_files(settings.DB_NORMALIZED_DIRECTORY)
    save_to_csv(filename, headers)

    for file in FILES:
        mesh = mesh_loader.load_mesh(file)
        splits = re.split('[\\\\/]', file)
        foldername = splits[-2]
        rectangularity = ed.calculate_rectangularity(mesh)
        compactness = ed.calculate_compactness(mesh)
        convexity = ed.calculate_convexity(mesh)
        eccentricity = ed.calculate_eccentricity(mesh)

        csv_data = [ 
            foldername,
            file,
            rectangularity,
            compactness,
            convexity,
            eccentricity
        ]
            
        save_to_csv(filename, csv_data)


def save_all_to_csv(filename: str, normalize: bool) -> None:
    helpers.create_output_folder(settings.ANALYZE_RESULTS_FOLDER)
    helpers.create_output_folder(settings.DB_NORMALIZED_DIRECTORY)
    FILES = helpers.list_db_files(settings.DB_ORIGINAL_DIRECTORY)

    headers = [ "shape class", "shape path", "face count", "vertex count", "is manifold", "bbox size", "bbox x position", "bbox y position", "bbox z position", "angle x axis", "angle y axis", "angle z axis" ]
    save_to_csv(filename, headers)
    i = 0
    for file in FILES:
        i+=1
        splits = re.split('[\\\\/]', file)
        meshname = splits[-1]
        foldername = splits[-2]

        mesh = mesh_loader.load_mesh(file, normalize=normalize)

        # XXX: Skipping non manifold mesh addition to DB csv when not manifold
        if mesh == None:
            continue

        if normalize:
            class_folder = f'{settings.DB_NORMALIZED_DIRECTORY}/{foldername}'
            helpers.create_output_folder(class_folder)

            file = f'{class_folder}/{meshname}'
            mesh.write(file)

        #Get all normalization data for analyzing the normalized meshes
        principle_axes, eigenvalues = rotation.calculate_principle_axes(mesh)
        axis_cos_x = rotation.get_vector_angle_cos((1, 0, 0), principle_axes[0])
        axis_cos_y = rotation.get_vector_angle_cos((0, 1, 0), principle_axes[1])
        axis_cos_z = rotation.get_vector_angle_cos((0, 0, 1), principle_axes[2])
        _, bbox_size, bbox_center, _ = scaling.compute_scaling_axis(mesh)
        bbox_pos = scaling.get_bbox_pos(bbox_center, mesh)

        csv_data = [ 
            foldername,
            file,
            len(mesh.cells()),
            len(mesh.points()), # type: ignore
            mesh.is_manifold(),
            bbox_size,
            *bbox_pos,
            axis_cos_x,
            axis_cos_y,
            axis_cos_z
        ]
            
        save_to_csv(filename, csv_data)
    print(i)

