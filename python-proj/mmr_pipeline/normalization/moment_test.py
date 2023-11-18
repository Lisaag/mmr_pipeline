import vedo
import numpy as np

def __get_triangle_center(p1, p2, p3):
    x = (p1[0] + p2[0] + p3[0]) / 3
    y = (p1[1] + p2[1] + p3[1]) / 3
    z = (p1[2] + p2[2] + p3[2]) / 3
    return (x, y, z)

# Perform a moment test, flipping the mesh to have more mass above the ground plane
def moment_test(mesh: vedo.Mesh, barycenter, principle_axes) -> None:
    updated_vertices = []
    f0 = 0
    f1 = 0
    f2 = 0
    for f in mesh.faces():
        A = mesh.points()[f][0]# type: ignore
        B = mesh.points()[f][1]# type: ignore
        C = mesh.points()[f][2]# type: ignore
        face_center = __get_triangle_center(A, B, C)

        f0 += np.sign(face_center[0]) * np.square(face_center[0])
        f1 += np.sign(face_center[1]) * np.square(face_center[1])
        f2 += np.sign(face_center[2]) * np.square(face_center[2])
    for p in mesh.points():# type: ignore
        
        new_point = [0, 0, 0]
        if f0 != 0: new_point[0] = p[0] * np.sign(f0)
        if f1 != 0: new_point[1] = p[1] * np.sign(f1)
        if f2 != 0: new_point[2] = p[2] * np.sign(f2)
        updated_vertices.append(new_point)

    mesh.points(updated_vertices) 

    