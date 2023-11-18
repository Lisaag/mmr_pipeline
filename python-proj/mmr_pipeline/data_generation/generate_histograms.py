import pandas as pd
import matplotlib.pyplot as plt
import settings as pipeline_settings, helpers
import numpy as np

def generate_label_chart(csv_file: str, target_column: str, filename: str, xlabel:str = "", ylabel:str = "", title:str = "") -> None:
    csv_df = pd.read_csv(csv_file)
    value_df = csv_df[target_column].value_counts()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    bar =plt.bar(range(len(value_df)), value_df.values, align='center', color = "#db6b6b") # type: ignore
    if target_column == 'shape class':
        #print(filename)
        values = bar.datavalues
        mean = np.mean(values)
        sd = np.std(values)
        # print(len(values))
        # print(mean)
        # print(sd)
        # print(np.max(values))
        # print(np.min(values))
    plt.xticks(range(len(value_df)), value_df.index.values, size=6.0, rotation=90) # type: ignore

    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f'{pipeline_settings.HISTOGRAM_OUTPUT_PATH}/{filename}')
    plt.clf()

def generate_histogram(csv_file: str, target_column: str | list[str], filename: str, bins: int | list | None = None, xlabel:str = "", ylabel:str = "", title:str = "") -> None:
    csv_df = pd.read_csv(csv_file)

    # if target_column == 'angle x axis' or target_column == 'angle y axis' or target_column == 'angle z axis':
    #     angles = csv_df.get(key=target_column)
    #     angles = angles.tolist()
    #     new_angles = [np.arccos(angle) * 180 /np.pi for angle in angles]
    #     print(new_angles)
    #     plt.hist(new_angles, bins=bins, color = "#db6b6b") # type: ignore
    # else:
    if filename == 'bbox_distance_normalized.png' or filename == 'bbox_distance_regular.png':
        x = csv_df.get(key='angle x axis')
        x = x.tolist()
        y = csv_df.get(key='angle y axis')
        y = y.tolist()
        z = csv_df.get(key='angle z axis')
        z = z.tolist()

        distances = []
        for i in range(len(x)):
            distance = np.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])
            #print(distance)
            distances.append(distance)
        plt.hist(distances, bins=bins, color = "#db6b6b") # type: ignore
    else: csv_df.hist(target_column, bins=bins, color = "#db6b6b") # type: ignore
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f'{pipeline_settings.HISTOGRAM_OUTPUT_PATH}/{filename}')
    plt.clf()

def generate_histograms() -> None:
    plt.rcParams['figure.dpi'] = 500
    helpers.create_output_folder(pipeline_settings.HISTOGRAM_OUTPUT_PATH)
    generate_label_chart(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'shape class', 'shape_class_regular.png',xlabel="Class labels", ylabel="Number of shapes", title="Shape distribution database" )
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'face count', 'face_count_regular.png', xlabel="Number of faces", ylabel="Number of shapes", title="Face count")
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'vertex count', 'vertex_count_regular.png', xlabel="Number of vertices", ylabel="Number of shapes", title="Vertex count")
    generate_label_chart(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'is manifold', 'manifold_regular.png', xlabel="Is manifold", ylabel="Number of shapes", title="Number of manifold meshes")
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'bbox size', 'bbox_size_regular.png', bins = [ i for i in range(0, 15)], xlabel="length bbox diagonal", ylabel="Number of shapes", title="Size object aligned bounding box")
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'bbox x position', 'bbox_x_regular.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'bbox y position', 'bbox_y_regular.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'bbox z position', 'bbox_z_regular.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'bbox z position', 'bbox_distance_regular.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'angle x axis', 'angle_x_axis_regular.png', xlabel="Cosine angle", ylabel="Number of shapes", title="Cosine angle major eigenvector and x-axis")
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'angle y axis', 'angle_y_axis_regular.png', xlabel="Cosine angle", ylabel="Number of shapes", title="Cosine angle medium eigenvector and y-axis")
    generate_histogram(pipeline_settings.CSV_UNNORMALIZED_OUTPUT_PATH, 'angle z axis', 'angle_z_axis_regular.png', xlabel="Cosine angle", ylabel="Number of shapes", title="Cosine angle minor eigenvector and z-axis")

    generate_label_chart(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'shape class', 'shape_class_normalized.png', xlabel="Class labels", ylabel="Number of shapes", title="Shape distribution database")
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'face count', 'face_count_normalized.png', xlabel="Number of faces", ylabel="Number of shapes", title="Face count")
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'vertex count', 'vertex_count_normalized.png', xlabel="Number of vertices", ylabel="Number of shapes", title="Vertex count")
    generate_label_chart(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'is manifold', 'manifold_normalized.png', xlabel="Is manifold", ylabel="Number of shapes", title="Number of manifold meshes")
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'bbox size', 'bbox_size_normalized.png', bins = [ i for i in range(0, 15)], xlabel="length bbox diagonal", ylabel="Number of shapes", title="Size object aligned bounding box")
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'bbox x position', 'bbox_x_normalized.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'bbox y position', 'bbox_y_normalized.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'bbox z position', 'bbox_z_normalized.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'bbox z position', 'bbox_distance_normalized.png', bins = [ i for i in range(-8, 8) ])
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'angle x axis', 'angle_x_axis_normalized.png', xlabel="Cosine angle", ylabel="Number of shapes", title="Cosine angle major eigenvector and x-axis")
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'angle y axis', 'angle_y_axis_normalized.png', xlabel="Cosine angle", ylabel="Number of shapes", title="Cosine angle medium eigenvector and y-axis")
    generate_histogram(pipeline_settings.CSV_NORMALIZED_OUTPUT_PATH, 'angle z axis', 'angle_z_axis_normalized.png', xlabel="Cosine angle", ylabel="Number of shapes", title="Cosine angle minor eigenvector and z-axis")


def generate_descriptor_histogram(raw_data, normalize_value, all_bins, samplecount = 1000, bincount = 30):

    fig, ax = plt.subplots()
    ax.set_autoscale_on(False)  
    
    ax.set_xlim(0, bincount) 
    ax.set_ylim(0, samplecount)  
    ax.set_xticks([0, bincount*0.2, bincount*0.4, bincount*0.6, bincount*0.8, 30])
    ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, samplecount*0.2, samplecount*0.4, samplecount*0.6, samplecount*0.8, 100])
    ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    #Plot a line for each shape of the same class
    for d in raw_data:
        if isinstance(d, str):
            data = np.array(d[1:-1].split(", ")).astype(np.float32)#[1:-1] to remove first and last char ([ ]), split to turn string into array
        else: data = d
        for i in range(len(data)):
            data[i] = data[i] / normalize_value

        line, bins, patches = ax.hist(data, bins=np.linspace(0.0, 1.0, bincount), histtype='step', alpha=0.0)

        plt.plot(line)

        line_normalized = np.array([i / samplecount for i in line])

        bin = line_normalized.tolist() #np.array(line).astype(np.int32)
        all_bins.append(bin)



        

def generate_all_descriptor_histograms(descriptor_name, normalize_value, samplecount=30, bincount = 30):
    csv_df = pd.read_csv(pipeline_settings.CSV_SHAPE_DESCRIPTORS_OUTPUT_PATH)
    shapeclass = csv_df.get(key="shapeclass")
    A3_descriptors = csv_df.get(key=descriptor_name)
    set_names = sorted(set(shapeclass)) #type: ignore

    all_bins = []

    all_A3_data = dict()
    for shape in set_names:
        all_A3_data[shape] = []

    #print(A3_descriptors) #type: ignore
    for i in range(len(A3_descriptors)):
        all_A3_data[shapeclass[i]] += [A3_descriptors[i]]

    for shape in set_names:
        generate_descriptor_histogram(all_A3_data[shape], normalize_value, all_bins)
        #TODO draw line foreach shape of the class
            # Add labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(shape + " - " + descriptor_name)
        plt.savefig(f'{pipeline_settings.HISTOGRAM_OUTPUT_PATH_DESCRIPTOR}/{descriptor_name}/{shape}')
        plt.close()

    return all_bins
