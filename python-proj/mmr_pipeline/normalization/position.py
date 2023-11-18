import vedo
import numpy as np

# Calculate the barycenter of a mesh
#
# XXX: Only calculates COM based on vert position, does not take faces into account
def calculate_barycenter(mesh: vedo.Mesh) -> np.ndarray:
    return mesh.GetCenter() # type: ignore

# Translate a mesh to the world origin based on its COM
def move_to_world_origin(mesh: vedo.Mesh) -> None:
    barycenter = calculate_barycenter(mesh)
    ORIGIN = np.array([0, 0, 0])
    translation = ORIGIN - barycenter

    mesh.SetPosition(*translation)
    mesh.SetOrigin(barycenter[0], barycenter[1], barycenter[2])
