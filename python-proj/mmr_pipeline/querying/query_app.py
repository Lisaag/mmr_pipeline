import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import vedo
import querying.feature_distance as fd
import scalability.clustering as clustering
import mesh_loader

#load mesh using vedo and restructure the vertex and face data, create subplot
def create_subplot(obj_file_path, isquery = False):
    mesh = vedo.Mesh(obj_file_path)
    vertices = np.array(mesh.points())
    vertices_x = [i[0] for i in vertices]
    vertices_y = [i[1] for i in vertices]
    vertices_z = [i[2] for i in vertices]
    faces = np.array(mesh.faces())
    faces_x = [i[0] for i in faces]
    faces_y = [i[1] for i in faces]
    faces_z = [i[2] for i in faces]

    color = '#AD00FF' if isquery else '#E2FF00'

    subplot = go.Mesh3d(
        x= vertices_x,
        y= vertices_y,
        z= vertices_z,
        i = faces_x,
        j = faces_y,
        k = faces_z,
        color=color,
        # showscale=False,
        # showlegend=False, 
        hoverinfo='skip',
        lighting=dict(ambient=0.3)
        )
    
   # subplot = dict(showgrid = 'xaxis',showticklabels = False)
    #subplot.scene = go.Scene(showgrid=False)

    return subplot

def query_similar_models():
    # Create a Streamlit file upload widget
    st.sidebar.header("Upload 3D OBJ File")
    check = st.sidebar.checkbox("k-NN query")

    query_k_NN = False
    #st.write('State of the checkbox: ', check)
    uploaded_file = st.sidebar.file_uploader("Choose a .obj file", type=["obj"])
    temp_write_loc = "temp.obj"

    if check:
        query_k_NN = True
        print(query_k_NN)
    if not check:
        query_k_NN = False
        print(query_k_NN)

    if uploaded_file:
        # Save the uploaded file to a temporary location
        with open(temp_write_loc, "wb") as f:
            f.write(uploaded_file.read())

        normalized_query_mesh = mesh_loader.load_mesh(temp_write_loc, True)
        normalized_query_mesh.write(temp_write_loc)
        if query_k_NN: k_nearest_shapes, k_nearest_distances, shape_classes = clustering.apply_knn(temp_write_loc)
        else: k_nearest_shapes, k_nearest_distances, shape_classes = fd.read_feature_vector(temp_write_loc)
        plot_titles = ["", "query_object", ""]
        for el in k_nearest_distances:
            plot_titles.append(("dis="+str(el)))

        # Create a Streamlit container for the Plotly figure
        figure_container = st.container()

        fig = make_subplots(rows=3, cols=3, row_heights=[200, 200, 200], column_widths=[100, 100, 100], specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}, {"type": "mesh3d"}],
                                                                                             [{"type": "mesh3d"}, {"type": "mesh3d"}, {"type": "mesh3d"}],
                                                                                             [{"type": "mesh3d"}, {"type": "mesh3d"}, {"type": "mesh3d"}]]
                                                                                             ,subplot_titles=plot_titles)
        plot1 = create_subplot(temp_write_loc, True)
        fig.add_trace(plot1, row=1, col=2)
        row_idx = 2

        for i in range(len(k_nearest_shapes)):
            plot = create_subplot(k_nearest_shapes[i])
            fig.add_trace(plot, row=row_idx, col=i%3+1)
            if(i%3 == 2): row_idx += 1

        layout = dict(xaxis = dict(nticks=4, range=[-.5, .5], visible=False),
                 yaxis = dict(nticks=4, range=[-.5, .5], visible=False),
                 zaxis = dict(nticks=4, range=[-.5, .5], visible=False),
                 aspectratio=dict(x=1.5, y=1.5, z=1.5))
        fig.update_layout(
            scene = layout, scene2 = layout, scene3 = layout,
            scene4 = layout, scene5 = layout, scene6 = layout,
            scene7 = layout, scene8 = layout, scene9 = layout
            , width=600, height=800)
        
        with figure_container:
            st.plotly_chart(fig)


        #fig.show()


def query_from_tsne(filename):
    k_nearest_shapes, k_nearest_distances, shape_classes = fd.read_feature_vector(filename)
    plot_titles = ["", "query_object", ""]
    for el in k_nearest_distances:
        plot_titles.append(("dis="+str(el)))

    # Create a Streamlit container for the Plotly figure
    figure_container = st.container()

    fig = make_subplots(rows=3, cols=3, row_heights=[200, 200, 200], column_widths=[100, 100, 100], specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}, {"type": "mesh3d"}],
                                                                                         [{"type": "mesh3d"}, {"type": "mesh3d"}, {"type": "mesh3d"}],
                                                                                         [{"type": "mesh3d"}, {"type": "mesh3d"}, {"type": "mesh3d"}]]
                                                                                         ,subplot_titles=plot_titles)
    plot1 = create_subplot(filename, True)
    fig.add_trace(plot1, row=1, col=2)
    row_idx = 2


    for i in range(len(k_nearest_shapes)):
        plot = create_subplot(k_nearest_shapes[i])
        fig.add_trace(plot, row=row_idx, col=i%3+1)
        if(i%3 == 2): row_idx += 1

    layout = dict(xaxis = dict(nticks=4, range=[-.5, .5], visible=False),
             yaxis = dict(nticks=4, range=[-.5, .5], visible=False),
             zaxis = dict(nticks=4, range=[-.5, .5], visible=False),
             aspectratio=dict(x=1.5, y=1.5, z=1.5))
    fig.update_layout(
        scene = layout, scene2 = layout, scene3 = layout,
        scene4 = layout, scene5 = layout, scene6 = layout,
        scene7 = layout, scene8 = layout, scene9 = layout
        , width=1000, height=1000)
    
    # with figure_container:
    #     st.plotly_chart(fig)


    fig.show()