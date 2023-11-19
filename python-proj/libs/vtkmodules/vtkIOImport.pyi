from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore

class vtk3DSCamera_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSCamera_t') -> None: ...

class vtk3DSChunk_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSChunk_t') -> None: ...

class vtk3DSColour_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSColour_t') -> None: ...

class vtk3DSColour_t_24(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSColour_t_24') -> None: ...

class vtk3DSFace_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSFace_t') -> None: ...

class vtkImporter(vtkmodules.vtkCommonCore.vtkObject):
    def DisableAnimation(self, animationIndex:int) -> None: ...
    def EnableAnimation(self, animationIndex:int) -> None: ...
    def GetAnimationName(self, animationIndex:int) -> str: ...
    def GetCameraName(self, camIndex:int) -> str: ...
    def GetNumberOfAnimations(self) -> int: ...
    def GetNumberOfCameras(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutputsDescription(self) -> str: ...
    def GetRenderWindow(self) -> 'vtkRenderWindow': ...
    def GetRenderer(self) -> 'vtkRenderer': ...
    def GetTemporalInformation(self, animationIndex:int, frameRate:float, nbTimeSteps:int, timeRange:MutableSequence[float], timeSteps:'vtkDoubleArray') -> bool: ...
    def IsA(self, type:str) -> int: ...
    def IsAnimationEnabled(self, animationIndex:int) -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkImporter': ...
    def Read(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkImporter': ...
    def SetCamera(self, camIndex:int) -> None: ...
    def SetRenderWindow(self, __a:'vtkRenderWindow') -> None: ...
    def Update(self) -> None: ...
    def UpdateTimeStep(self, timeValue:float) -> None: ...

class vtk3DSImporter(vtkImporter):
    def ComputeNormalsOff(self) -> None: ...
    def ComputeNormalsOn(self) -> None: ...
    def GetComputeNormals(self) -> int: ...
    def GetFileName(self) -> str: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutputsDescription(self) -> str: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtk3DSImporter': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtk3DSImporter': ...
    def SetComputeNormals(self, _arg:int) -> None: ...
    def SetFileName(self, _arg:str) -> None: ...

class vtk3DSList_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSList_t') -> None: ...

class vtk3DSMatProp_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSMatProp_t') -> None: ...

class vtk3DSMaterial_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSMaterial_t') -> None: ...

class vtk3DSMesh_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSMesh_t') -> None: ...

class vtk3DSOmniLight_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSOmniLight_t') -> None: ...

class vtk3DSSpotLight_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSSpotLight_t') -> None: ...

class vtk3DSSummary_t(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtk3DSSummary_t') -> None: ...

class vtkGLTFImporter(vtkImporter):
    def DisableAnimation(self, animationIndex:int) -> None: ...
    def EnableAnimation(self, animationIndex:int) -> None: ...
    def GetAnimationName(self, animationIndex:int) -> str: ...
    def GetCamera(self, id:int) -> 'vtkCamera': ...
    def GetCameraName(self, camIndex:int) -> str: ...
    def GetFileName(self) -> str: ...
    def GetNumberOfAnimations(self) -> int: ...
    def GetNumberOfCameras(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutputsDescription(self) -> str: ...
    def GetTemporalInformation(self, animationIndex:int, frameRate:float, nbTimeSteps:int, timeRange:MutableSequence[float], timeSteps:'vtkDoubleArray') -> bool: ...
    def IsA(self, type:str) -> int: ...
    def IsAnimationEnabled(self, animationIndex:int) -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkGLTFImporter': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkGLTFImporter': ...
    def SetCamera(self, camIndex:int) -> None: ...
    def SetFileName(self, _arg:str) -> None: ...
    def UpdateTimeStep(self, timeValue:float) -> None: ...

class vtkOBJImporter(vtkImporter):
    def GetFileName(self) -> str: ...
    def GetFileNameMTL(self) -> str: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutputDescription(self, idx:int) -> str: ...
    def GetOutputsDescription(self) -> str: ...
    def GetTexturePath(self) -> str: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkOBJImporter': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkOBJImporter': ...
    def SetFileName(self, arg:str) -> None: ...
    def SetFileNameMTL(self, arg:str) -> None: ...
    def SetTexturePath(self, path:str) -> None: ...

class vtkVRMLImporter(vtkImporter):
    def GetFileName(self) -> str: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOutputsDescription(self) -> str: ...
    def GetShapeResolution(self) -> int: ...
    def GetVRMLDEFObject(self, name:str) -> 'vtkObject': ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkVRMLImporter': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkVRMLImporter': ...
    def SetFileName(self, _arg:str) -> None: ...
    def SetShapeResolution(self, _arg:int) -> None: ...
