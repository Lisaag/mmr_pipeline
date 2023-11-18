import vedo
import numpy as np
import normalization.scaling as scaling
import normalization.rotation as rotation
import copy

def calculate_rectangularity(mesh: vedo.Mesh) -> float:
    axis_length, _, _, bbox_volume = scaling.compute_scaling_axis(mesh)
    #obb_volume = axis_length ** 3

    return mesh.volume() / bbox_volume

def calculate_compactness(mesh: vedo.Mesh) -> float:
    mesh_volume = mesh.volume()
    mesh_area = mesh.area()

    return np.power(mesh_area, 3)/(36.0 * np.pi * np.power(mesh_volume, 2))

def calculate_spherecity(mesh: vedo.Mesh) -> float:
    return 1 / calculate_compactness(mesh)

#convexity of the shape: "shape volume divided by convex hull volume"
#value 1 = completely convex, value closer to 0 = very concave
def calculate_convexity(mesh: vedo.Mesh) -> float:
    convex_hull = vedo.ConvexHull(mesh.clone().points())
    convex_hull_volume = convex_hull.volume()
    mesh_volume = mesh.volume()
    convexity = 0
    if convex_hull_volume == 0: convexity = convex_hull_volume
    else: convexity = mesh_volume / convex_hull_volume

    return convexity

def calculate_eccentricity(mesh: vedo.Mesh) -> float:
    principal_axes, eigenvalues = rotation.calculate_principle_axes(mesh)
    eccentricity = eigenvalues[0] / eigenvalues[2]

    return eccentricity

