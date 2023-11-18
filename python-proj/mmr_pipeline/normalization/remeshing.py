import settings
import vedo
import vedo.pointcloud as pc

#Fix faulty meshes
def fix_mesh(mesh: vedo.Mesh) -> None:
    #print("fix")
    mesh.triangulate()
    try:
        mesh.non_manifold_faces(remove=True, tol='auto')
    except:
        print("CANNOT GET RID OF NON-MANIFOLD FACES")
        return
    if not mesh.is_closed():
        mesh.fill_holes(size=0.5)



# Remesh a mesh with either subdivision or decimaxtion based on its vertex count, and the target vertex count set in settings
def remesh(mesh: vedo.Mesh) -> None:
    if len(mesh.points()) < (settings.MESH_VERTEX_COUNT - settings.MESH_VERTEX_EPSILON):# type: ignore
        mesh.subdivide(
            2,
            3
        )  
    elif len(mesh.points()) > (settings.MESH_VERTEX_COUNT + settings.MESH_VERTEX_EPSILON):# type: ignore
        try:
            mesh.decimate(fraction=0.8, n=settings.MESH_VERTEX_COUNT)
        except Exception:
            return
    mesh.compute_normals(consistency=True) 