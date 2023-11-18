import vedo
import math
import numpy as np

# Calculate the magnitude of an input vector
def __magnitude(vector) -> int | float:
    return math.sqrt(sum(pow(element, 2) for element in vector))

def get_vector_angle_cos(v1, v2):
    mag1 = __magnitude(v1)
    mag2 = __magnitude(v2)
    dot = np.dot(v1, v2)
    cos_angle = dot / (mag1 * mag2)
    return cos_angle
# Calculate principle axes for a mesh based on its point cloud
#
# Returns eigen vectors indicating x, y, z axes for mesh
def calculate_principle_axes(mesh: vedo.Mesh) -> tuple[list[np.ndarray], list[float]]:
    vertices = np.asarray(mesh.points())

    cov = np.cov(vertices.transpose())
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    eigencombined = [(eigenvalues[i], eigenvectors[:, i]) for i in range(3)]
    eigencombined.sort(key=lambda x:x[0], reverse=True)

    eigenvectors = [item[1] for item in eigencombined]
    eigenvalues = [item[0] for item in eigencombined]

    eigenvectors.pop(2)
    eigenvectors.append(np.cross(eigenvectors[0], eigenvectors[1])) 

    return eigenvectors, eigenvalues

# Normalize the mesh pose using principle axes, aligning it with the world coordinate system
def normalize_pose(mesh: vedo.Mesh, principle_axes: list) -> None:
    major_principle_axis = principle_axes[0]
    medium_principle_axis = principle_axes[1]

    updated_vertex_positions = []
    for p in mesh.points(): #type: ignore
        x = np.dot(p, major_principle_axis)
        y = np.dot(p, medium_principle_axis)
        z = np.dot(p, np.cross(major_principle_axis, medium_principle_axis))
        p = (x, y, z)
        updated_vertex_positions.append(p)
    mesh.points(updated_vertex_positions)

