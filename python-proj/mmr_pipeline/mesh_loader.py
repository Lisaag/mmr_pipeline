import vedo
import pandas as pd
import settings, helpers
from normalization import remeshing, position, rotation, moment_test, scaling


# Apply mesh triangulation if input mesh is not triangulated
def try_triangulate(mesh: vedo.Mesh) -> vedo.Mesh:
    is_triangulated = all(len(face) == 3 for face in mesh.faces())

    if not is_triangulated:
        mesh.triangulate()

    return mesh

# Load a mesh by filepath
# Applies triangulation and normalization if needed
def load_mesh(filepath: str, normalize: bool = False) -> vedo.Mesh:
    mesh = vedo.Mesh(filepath)
    mesh.color('white') # type: ignore

    if (normalize):
        #TODO Fixing up the mesh; stitching holes and <- still needs to work, maybe use PyMeshLab
        remeshing.fix_mesh(mesh)

        #TODO if mesh is still manifold after fixing, just remove from database <--Now skips all non-manifold meshes
       # if mesh.is_manifold():
        # Remesh mesh to normalized vertex count
        try:
            counter = 0
            while(len(mesh.points()) > (settings.MESH_VERTEX_COUNT + settings.MESH_VERTEX_EPSILON) or len(mesh.points())  < (settings.MESH_VERTEX_COUNT - settings.MESH_VERTEX_EPSILON)): # type: ignore
                remeshing.remesh(mesh)
                counter += 1
                if(counter > 50):
                    raise Exception("stuck in remeshing")

            # Normalize position
            position.move_to_world_origin(mesh)

            # Calculate principle axes for use in moment test (can also be done inside that function?)
            principle_axes, eigenvalues = rotation.calculate_principle_axes(mesh)

            # Normalize rotation
            rotation.normalize_pose(mesh, principle_axes)

            # Perform a moment test to possibly flip the mesh
            moment_test.moment_test(mesh, position.calculate_barycenter(mesh), principle_axes)

            # Scale the mesh to unit size
            scaling.normalize_scale(mesh)

            mesh.computeNormals()
        except Exception:
            print("Something went wrong with sub/super division")
            return None
        #else: return None
    if normalize: 
        if(len(mesh.points()) > (settings.MESH_VERTEX_COUNT + settings.MESH_VERTEX_EPSILON) or len(mesh.points())  < (settings.MESH_VERTEX_COUNT - settings.MESH_VERTEX_EPSILON)): # type: ignore
            return None
    return mesh
  
# Display all meshes one by one
def display_all(normalize: bool) -> None:
    FILES = helpers.list_db_files(settings.DB_ORIGINAL_DIRECTORY)
    display_list(FILES, normalize)

def display_list(files: list[str], normalize: bool) -> None:
    for file in files:
        display(file, normalize)

def display_csv(csv_path: str, normalize: bool) -> None:
    data = pd.read_csv(csv_path)
    FILES = data['shape path']

    display_list(list(FILES), normalize)

def display(file: str, normalize: bool) -> None:
    # Load the mesh from file
    mesh = load_mesh(file, normalize)
    if(not mesh == None): helpers.show_mesh_plot(mesh, show_unit_cube=normalize)
