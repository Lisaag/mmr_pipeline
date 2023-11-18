import vedo
import feature_extraction.elementary_descriptors as ed
import helpers

def get_descr(mesh: vedo.Mesh):
    rect = ed.calculate_rectangularity(mesh)
    comp = ed.calculate_compactness(mesh)
    conv = ed.calculate_convexity(mesh)
    ecc = ed.calculate_eccentricity(mesh)

    print(rect, comp, conv, ecc)

def get_descriptors_primary_shape():
    sphere = vedo.Sphere()
    sphere.triangulate()
    hull = vedo.ConvexHull(sphere.points())
    helpers.show_mesh_plot(hull, show_unit_cube=False)
    get_descr(sphere)
    cube = vedo.Box(width=1, height=.1, length=.1)
    cube.triangulate()
    helpers.show_mesh_plot(cube, show_unit_cube=False)
    #helpers.show_mesh_plot(cube, show_unit_cube=False)
    get_descr(cube)






