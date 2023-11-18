import vedo
from feature_extraction import elementary_descriptors
import mesh_loader

def extract_features(mesh: vedo.Mesh) -> dict:
    return {
        'spherecity': elementary_descriptors.calculate_spherecity(mesh),
        'rectangularity': elementary_descriptors.calculate_rectangularity(mesh),
        'convexity': elementary_descriptors.calculate_convexity(mesh)
    }
