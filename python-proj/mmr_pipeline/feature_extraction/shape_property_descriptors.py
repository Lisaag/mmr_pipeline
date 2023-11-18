import vedo
import numpy as np
import random
import math

SAMPLE_COUNT = 1000

def __magnitude(vector) -> int | float:
    return math.sqrt(sum(pow(element, 2) for element in vector))

#normalize de lengte van de vectoren eerst!!
def __get_angle(v1, v2) -> float:
    # v1 = np.linalg.norm(v1)
    # v2 = np.linalg.norm(v2)
    mag1 = __magnitude(v1)# type: ignore
    mag2 = __magnitude(v2)# type: ignore
    if(mag1 < 0.000001 or np.isnan(mag1)): 
        print(v1)
        return 0.0 
    if(mag2 < 0.000001 or np.isnan(mag2)):
        print(v2)
        return 0.0 
    v1 = v1 / mag1
    v2 = v2 / mag2
    mag1 = __magnitude(v1)# type: ignore
    mag2 = __magnitude(v2)# type: ignore
    cos_angle = np.dot(v1, v2) # No division by magnitude, bc vectors are normalized / (mag1 * mag2)
    if(cos_angle < -1.0 or cos_angle > 1.0):
        print("DOT")
        print(np.dot(v1, v2))
        print(mag1)
        print(mag2)
        return 0.0

    angle = np.arccos(cos_angle) # type: ignore
    return angle

def shape_property_a3(mesh: vedo.Mesh) -> list[float]:
    angles: list[float] = []
    vertex_count = len(mesh.points())#type: ignore
    random_indices = random.sample(range(0, vertex_count), SAMPLE_COUNT * 3)

    for i in range(SAMPLE_COUNT):
        p1 = mesh.points()[random_indices[i * 3]]#type: ignore
        p2 = mesh.points()[random_indices[i * 3 + 1]]#type: ignore
        p3 = mesh.points()[random_indices[i * 3 + 2]]#type: ignore

        v1 = p1 - p2
        v2 = p1 - p3

        angle_degrees = __get_angle(v1, v2) * (180.0 / np.pi)
        angles.append(angle_degrees)

    return angles

def shape_property_d1(mesh: vedo.Mesh) -> list[float] :
    distances = []
    barycenter = mesh.GetCenter()
    vertex_count = len(mesh.points())#type: ignore
    random_indices = random.sample(range(0, vertex_count), SAMPLE_COUNT)

    for i in range(SAMPLE_COUNT):
        v = barycenter - mesh.points()[random_indices[i]] #type: ignore
        distance = __magnitude(v) #type: ignore
        distances.append(distance)

    return distances

def shape_property_d2(mesh: vedo.Mesh) -> list[float] :
    distances = []
    vertex_count = len(mesh.points())#type: ignore
    random_indices = random.sample(range(0, vertex_count), SAMPLE_COUNT * 2)

    for i in range(SAMPLE_COUNT):
        p1 = mesh.points()[random_indices[i * 2]] #type: ignore
        p2 = mesh.points()[random_indices[i * 2 + 1]] #type: ignore
        v = p1 - p2
        distance = __magnitude(v) #type: ignore
        distances.append(distance)

    return distances

#https://www.wikihow.com/Calculate-the-Area-of-a-Triangle <- I just used method 4 from this link
def shape_property_d3(mesh: vedo.Mesh) -> list[float] :
    areas: list[float] = []
    vertex_count = len(mesh.points())#type: ignore
    random_indices = random.sample(range(0, vertex_count), SAMPLE_COUNT * 3)

    for i in range(SAMPLE_COUNT):
        p1 = mesh.points()[random_indices[i * 3]]#type: ignore
        p2 = mesh.points()[random_indices[i * 3 + 1]]#type: ignore
        p3 = mesh.points()[random_indices[i * 3 + 2]]#type: ignore

        v1 = p1 - p2
        v2 = p1 - p3
        mag1 = __magnitude(v1) #type:ignore
        mag2 = __magnitude(v2) #type:ignore

        angle = __get_angle(v1, v2)
        area = ((mag1 * mag2) / 2.0) * np.sin(angle)

        #take the square root of the area
        area = np.sqrt(area)

        areas.append(area)

    return areas

def shape_property_d4(mesh: vedo.Mesh) -> list[float] :
    volumes: list[float] = []
    vertex_count = len(mesh.points())#type: ignore
    random_indices = random.sample(range(0, vertex_count), SAMPLE_COUNT * 4)

    for i in range(SAMPLE_COUNT):
        p1 = mesh.points()[random_indices[i * 4]]#type: ignore
        p2 = mesh.points()[random_indices[i * 4 + 1]]#type: ignore
        p3 = mesh.points()[random_indices[i * 4 + 2]]#type: ignore
        p4 = mesh.points()[random_indices[i * 4 + 3]]#type: ignore

        v1 = p1 - p2
        v2 = p1 - p3
        mag1 = __magnitude(v1) #type:ignore
        mag2 = __magnitude(v2) #type:ignore

        angle = __get_angle(v1, v2)
        area = ((mag1 * mag2) / 2.0) * np.sin(angle)

        n = np.cross(v1, v2)
        height = np.abs(np.dot((p1-p4), n) / __magnitude(n))
        
        volume = (1.0/3.0) * area * height

        #take the cube root of the volume
        volume = np.cbrt(volume)

        volumes.append(volume)

    return volumes