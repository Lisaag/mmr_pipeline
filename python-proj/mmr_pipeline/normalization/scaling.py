import vedo
import numpy as np
from normalization import position

#get the largest local axis to determine the scaling factor
def compute_scaling_axis(mesh: vedo.Mesh) -> tuple[float, float, list[float], float]:
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []

    for p in mesh.points(): # type: ignore
        x.append(p[0])
        y.append(p[1])
        z.append(p[2])

    min_x: float = min(x)   
    min_y: float = min(y)   
    min_z: float = min(z)   

    max_x: float = max(x)   
    max_y: float = max(y)   
    max_z: float = max(z) 

    max_axis: float = max([max_x - min_x, max_y - min_y, max_z - min_z])  
    bbox_diagonal: float = np.sqrt(np.absolute(np.dot([max_axis, max_axis, max_axis], [max_axis, max_axis, max_axis])))
    bbox_center: list[float] = [ max_axis / 2.0 for i in range(3) ]

    bbox_volume: float = np.abs(max_x - min_x) * np.abs(max_y - min_y) * np.abs(max_z - min_z)
    
    return max_axis, bbox_diagonal, bbox_center, bbox_volume

# Calculate bounding box world space position using a mesh and the bounding box center
def get_bbox_pos(bbox_center: list[float], mesh: vedo.Mesh) -> list[float]:
    bbox_pos: list[float] = [0, 0, 0]
    mesh_position = position.calculate_barycenter(mesh)

    bbox_pos[0] = bbox_center[0] - mesh_position[0]
    bbox_pos[1] = bbox_center[1] - mesh_position[1]
    bbox_pos[2] = bbox_center[2] - mesh_position[2]

    return bbox_pos

#Set new scale for each vertex (maybe we can also just doe mesh.SetScale(), but I just did it as in the technical tips docs)
#Returns the diagonal of the bounding box, because we will store this in csv
def normalize_scale(mesh: vedo.Mesh) -> None:
    max_axis, _, _, _ = compute_scaling_axis(mesh)

    scale_factor = 1 / max_axis

    updated_vertex_positions = []
    for p in mesh.points(): # type: ignore
        updated_pos = scale_factor * p
        updated_vertex_positions.append(updated_pos)

    mesh.points(updated_vertex_positions)
    