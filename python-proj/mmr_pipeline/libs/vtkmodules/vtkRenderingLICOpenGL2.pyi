from typing import overload, Any, Callable, TypeVar, Union
from typing import Tuple, List, Sequence, MutableSequence

Callback = Union[Callable[..., None], None]
Buffer = TypeVar('Buffer')
Pointer = TypeVar('Pointer')
Template = TypeVar('Template')

import vtkmodules.vtkCommonCore
import vtkmodules.vtkCommonExecutionModel
import vtkmodules.vtkRenderingCore
import vtkmodules.vtkRenderingOpenGL2

class vtkBatchedSurfaceLICMapper(vtkmodules.vtkRenderingOpenGL2.vtkOpenGLBatchedPolyDataMapper):
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkBatchedSurfaceLICMapper': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkBatchedSurfaceLICMapper': ...

class vtkCompositeSurfaceLICMapper(vtkmodules.vtkRenderingCore.vtkCompositePolyDataMapper):
    def GetLICInterface(self) -> 'vtkSurfaceLICInterface': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositeSurfaceLICMapper': ...
    def Render(self, ren:'vtkRenderer', act:'vtkActor') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositeSurfaceLICMapper': ...

class vtkCompositeSurfaceLICMapperDelegator(vtkmodules.vtkRenderingOpenGL2.vtkOpenGLCompositePolyDataMapperDelegator):
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkCompositeSurfaceLICMapperDelegator': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkCompositeSurfaceLICMapperDelegator': ...
    def ShallowCopy(self, mapper:'vtkCompositePolyDataMapper') -> None: ...

class vtkImageDataLIC2D(vtkmodules.vtkCommonExecutionModel.vtkImageAlgorithm):
    def GetContext(self) -> 'vtkRenderWindow': ...
    def GetMagnification(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetOpenGLExtensionsSupported(self) -> int: ...
    def GetStepSize(self) -> float: ...
    def GetSteps(self) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkImageDataLIC2D': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkImageDataLIC2D': ...
    def SetContext(self, context:'vtkRenderWindow') -> int: ...
    def SetMagnification(self, _arg:int) -> None: ...
    def SetStepSize(self, _arg:float) -> None: ...
    def SetSteps(self, _arg:int) -> None: ...
    def TranslateInputExtent(self, inExt:Sequence[int], inWholeExtent:Sequence[int], outExt:MutableSequence[int]) -> None: ...

class vtkLineIntegralConvolution2D(vtkmodules.vtkCommonCore.vtkObject):
    ENHANCE_CONTRAST_OFF:int
    ENHANCE_CONTRAST_ON:int
    def AntiAliasOff(self) -> None: ...
    def AntiAliasOn(self) -> None: ...
    def EnhanceContrastOff(self) -> None: ...
    def EnhanceContrastOn(self) -> None: ...
    def EnhancedLICOff(self) -> None: ...
    def EnhancedLICOn(self) -> None: ...
    @overload
    def Execute(self, vectorTex:'vtkTextureObject', noiseTex:'vtkTextureObject') -> 'vtkTextureObject': ...
    @overload
    def Execute(self, extent:Sequence[int], vectorTex:'vtkTextureObject', noiseTex:'vtkTextureObject') -> 'vtkTextureObject': ...
    def GetAntiAlias(self) -> int: ...
    def GetAntiAliasMaxValue(self) -> int: ...
    def GetAntiAliasMinValue(self) -> int: ...
    def GetComponentIds(self) -> Tuple[int, int]: ...
    def GetContext(self) -> 'vtkOpenGLRenderWindow': ...
    def GetEnhanceContrast(self) -> int: ...
    def GetEnhanceContrastMaxValue(self) -> int: ...
    def GetEnhanceContrastMinValue(self) -> int: ...
    def GetEnhancedLIC(self) -> int: ...
    def GetEnhancedLICMaxValue(self) -> int: ...
    def GetEnhancedLICMinValue(self) -> int: ...
    def GetHighContrastEnhancementFactor(self) -> float: ...
    def GetHighContrastEnhancementFactorMaxValue(self) -> float: ...
    def GetHighContrastEnhancementFactorMinValue(self) -> float: ...
    def GetLowContrastEnhancementFactor(self) -> float: ...
    def GetLowContrastEnhancementFactorMaxValue(self) -> float: ...
    def GetLowContrastEnhancementFactorMinValue(self) -> float: ...
    def GetMaskThreshold(self) -> float: ...
    def GetMaskThresholdMaxValue(self) -> float: ...
    def GetMaskThresholdMinValue(self) -> float: ...
    def GetMaxNoiseValue(self) -> float: ...
    def GetMaxNoiseValueMaxValue(self) -> float: ...
    def GetMaxNoiseValueMinValue(self) -> float: ...
    def GetNormalizeVectors(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfSteps(self) -> int: ...
    def GetNumberOfStepsMaxValue(self) -> int: ...
    def GetNumberOfStepsMinValue(self) -> int: ...
    def GetStepSize(self) -> float: ...
    def GetStepSizeMaxValue(self) -> float: ...
    def GetStepSizeMinValue(self) -> float: ...
    def GetTransformVectors(self) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsSupported(renWin:'vtkRenderWindow') -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkLineIntegralConvolution2D': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkLineIntegralConvolution2D': ...
    def SetAntiAlias(self, _arg:int) -> None: ...
    @overload
    def SetComponentIds(self, c0:int, c1:int) -> None: ...
    @overload
    def SetComponentIds(self, c:MutableSequence[int]) -> None: ...
    def SetContext(self, context:'vtkOpenGLRenderWindow') -> None: ...
    def SetEnhanceContrast(self, _arg:int) -> None: ...
    def SetEnhancedLIC(self, _arg:int) -> None: ...
    def SetHighContrastEnhancementFactor(self, _arg:float) -> None: ...
    def SetLowContrastEnhancementFactor(self, _arg:float) -> None: ...
    def SetMaskThreshold(self, _arg:float) -> None: ...
    def SetMaxNoiseValue(self, _arg:float) -> None: ...
    @staticmethod
    def SetNoiseTexParameters(noise:'vtkTextureObject') -> None: ...
    def SetNormalizeVectors(self, val:int) -> None: ...
    def SetNumberOfSteps(self, _arg:int) -> None: ...
    def SetStepSize(self, _arg:float) -> None: ...
    def SetTransformVectors(self, val:int) -> None: ...
    @staticmethod
    def SetVectorTexParameters(vectors:'vtkTextureObject') -> None: ...
    def WriteTimerLog(self, __a:str) -> None: ...

class vtkPainterCommunicator(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, other:'vtkPainterCommunicator') -> None: ...
    def GetIsNull(self) -> bool: ...
    def GetMPIFinalized(self) -> bool: ...
    def GetMPIInitialized(self) -> bool: ...
    def GetRank(self) -> int: ...
    def GetSize(self) -> int: ...
    def GetWorldRank(self) -> int: ...
    def GetWorldSize(self) -> int: ...

class vtkStructuredGridLIC2D(vtkmodules.vtkCommonExecutionModel.vtkStructuredGridAlgorithm):
    def GetContext(self) -> 'vtkRenderWindow': ...
    def GetFBOSuccess(self) -> int: ...
    def GetLICSuccess(self) -> int: ...
    def GetMagnification(self) -> int: ...
    def GetMagnificationMaxValue(self) -> int: ...
    def GetMagnificationMinValue(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetStepSize(self) -> float: ...
    def GetSteps(self) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkStructuredGridLIC2D': ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkStructuredGridLIC2D': ...
    def SetContext(self, context:'vtkRenderWindow') -> int: ...
    def SetMagnification(self, _arg:int) -> None: ...
    def SetStepSize(self, _arg:float) -> None: ...
    def SetSteps(self, _arg:int) -> None: ...

class vtkSurfaceLICComposite(vtkmodules.vtkCommonCore.vtkObject):
    COMPOSITE_AUTO:int
    COMPOSITE_BALANCED:int
    COMPOSITE_INPLACE:int
    COMPOSITE_INPLACE_DISJOINT:int
    def BuildProgram(self, __a:MutableSequence[float]) -> int: ...
    def GetCompositeExtent(self, i:int=0) -> 'vtkPixelExtent': ...
    def GetContext(self) -> 'vtkOpenGLRenderWindow': ...
    def GetDataSetExtent(self) -> 'vtkPixelExtent': ...
    def GetDisjointGuardExtent(self, i:int=0) -> 'vtkPixelExtent': ...
    def GetGuardExtent(self, i:int=0) -> 'vtkPixelExtent': ...
    def GetNumberOfCompositeExtents(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetStrategy(self) -> int: ...
    def GetWindowExtent(self) -> 'vtkPixelExtent': ...
    def InitializeCompositeExtents(self, vectors:MutableSequence[float]) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSurfaceLICComposite': ...
    def RestoreDefaultCommunicator(self) -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSurfaceLICComposite': ...
    def SetContext(self, __a:'vtkOpenGLRenderWindow') -> None: ...
    def SetStrategy(self, val:int) -> None: ...

class vtkSurfaceLICInterface(vtkmodules.vtkCommonCore.vtkObject):
    COLOR_MODE_BLEND:int
    COLOR_MODE_MAP:int
    COMPOSITE_AUTO:int
    COMPOSITE_BALANCED:int
    COMPOSITE_INPLACE:int
    COMPOSITE_INPLACE_DISJOINT:int
    ENHANCE_CONTRAST_BOTH:int
    ENHANCE_CONTRAST_COLOR:int
    ENHANCE_CONTRAST_LIC:int
    ENHANCE_CONTRAST_OFF:int
    NOISE_TYPE_GAUSSIAN:int
    NOISE_TYPE_PERLIN:int
    NOISE_TYPE_UNIFORM:int
    def AntiAliasOff(self) -> None: ...
    def AntiAliasOn(self) -> None: ...
    def ApplyLIC(self) -> None: ...
    def CanRenderSurfaceLIC(self, actor:'vtkActor') -> bool: ...
    def CombineColorsAndLIC(self) -> None: ...
    def CompletedGeometry(self) -> None: ...
    def CopyToScreen(self) -> None: ...
    def CreateCommunicator(self, __a:'vtkRenderer', __b:'vtkActor', data:'vtkDataObject') -> None: ...
    def EnableOff(self) -> None: ...
    def EnableOn(self) -> None: ...
    def EnhancedLICOff(self) -> None: ...
    def EnhancedLICOn(self) -> None: ...
    def GatherVectors(self) -> None: ...
    def GetAntiAlias(self) -> int: ...
    def GetColorMode(self) -> int: ...
    def GetCompositeStrategy(self) -> int: ...
    def GetEnable(self) -> int: ...
    def GetEnhanceContrast(self) -> int: ...
    def GetEnhancedLIC(self) -> int: ...
    def GetGenerateNoiseTexture(self) -> int: ...
    def GetHasVectors(self) -> bool: ...
    def GetHighColorContrastEnhancementFactor(self) -> float: ...
    def GetHighLICContrastEnhancementFactor(self) -> float: ...
    def GetImpulseNoiseBackgroundValue(self) -> float: ...
    def GetImpulseNoiseProbability(self) -> float: ...
    def GetLICIntensity(self) -> float: ...
    def GetLowColorContrastEnhancementFactor(self) -> float: ...
    def GetLowLICContrastEnhancementFactor(self) -> float: ...
    def GetMapModeBias(self) -> float: ...
    def GetMaskColor(self) -> Tuple[float, float, float]: ...
    def GetMaskIntensity(self) -> float: ...
    def GetMaskOnSurface(self) -> int: ...
    def GetMaskThreshold(self) -> float: ...
    def GetMaxNoiseValue(self) -> float: ...
    def GetMinNoiseValue(self) -> float: ...
    def GetNoiseDataSet(self) -> 'vtkImageData': ...
    def GetNoiseGeneratorSeed(self) -> int: ...
    def GetNoiseGrainSize(self) -> int: ...
    def GetNoiseTextureSize(self) -> int: ...
    def GetNoiseType(self) -> int: ...
    def GetNormalizeVectors(self) -> int: ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def GetNumberOfNoiseLevels(self) -> int: ...
    def GetNumberOfSteps(self) -> int: ...
    def GetStepSize(self) -> float: ...
    def InitializeResources(self) -> None: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsSupported(context:'vtkRenderWindow') -> bool: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def MaskOnSurfaceOff(self) -> None: ...
    def MaskOnSurfaceOn(self) -> None: ...
    def NewInstance(self) -> 'vtkSurfaceLICInterface': ...
    def NormalizeVectorsOff(self) -> None: ...
    def NormalizeVectorsOn(self) -> None: ...
    def PrepareForGeometry(self) -> None: ...
    def ReleaseGraphicsResources(self, win:'vtkWindow') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSurfaceLICInterface': ...
    def SetAntiAlias(self, val:int) -> None: ...
    def SetColorMode(self, val:int) -> None: ...
    def SetCompositeStrategy(self, val:int) -> None: ...
    def SetEnable(self, _arg:int) -> None: ...
    def SetEnhanceContrast(self, val:int) -> None: ...
    def SetEnhancedLIC(self, val:int) -> None: ...
    def SetGenerateNoiseTexture(self, shouldGenerate:int) -> None: ...
    def SetHasVectors(self, val:bool) -> None: ...
    def SetHighColorContrastEnhancementFactor(self, val:float) -> None: ...
    def SetHighLICContrastEnhancementFactor(self, val:float) -> None: ...
    def SetImpulseNoiseBackgroundValue(self, val:float) -> None: ...
    def SetImpulseNoiseProbability(self, val:float) -> None: ...
    def SetLICIntensity(self, val:float) -> None: ...
    def SetLowColorContrastEnhancementFactor(self, val:float) -> None: ...
    def SetLowLICContrastEnhancementFactor(self, val:float) -> None: ...
    def SetMapModeBias(self, val:float) -> None: ...
    @overload
    def SetMaskColor(self, val:MutableSequence[float]) -> None: ...
    @overload
    def SetMaskColor(self, r:float, g:float, b:float) -> None: ...
    def SetMaskIntensity(self, val:float) -> None: ...
    def SetMaskOnSurface(self, val:int) -> None: ...
    def SetMaskThreshold(self, val:float) -> None: ...
    def SetMaxNoiseValue(self, val:float) -> None: ...
    def SetMinNoiseValue(self, val:float) -> None: ...
    def SetNoiseDataSet(self, data:'vtkImageData') -> None: ...
    def SetNoiseGeneratorSeed(self, val:int) -> None: ...
    def SetNoiseGrainSize(self, val:int) -> None: ...
    def SetNoiseTextureSize(self, length:int) -> None: ...
    def SetNoiseType(self, type:int) -> None: ...
    def SetNormalizeVectors(self, val:int) -> None: ...
    def SetNumberOfNoiseLevels(self, val:int) -> None: ...
    def SetNumberOfSteps(self, val:int) -> None: ...
    def SetStepSize(self, val:float) -> None: ...
    def ShallowCopy(self, m:'vtkSurfaceLICInterface') -> None: ...
    def UpdateCommunicator(self, renderer:'vtkRenderer', actor:'vtkActor', data:'vtkDataObject') -> None: ...
    def ValidateContext(self, renderer:'vtkRenderer') -> None: ...
    def WriteTimerLog(self, __a:str) -> None: ...

class vtkSurfaceLICMapper(vtkmodules.vtkRenderingOpenGL2.vtkOpenGLPolyDataMapper):
    def GetLICInterface(self) -> 'vtkSurfaceLICInterface': ...
    def GetNumberOfGenerationsFromBase(self, type:str) -> int: ...
    @staticmethod
    def GetNumberOfGenerationsFromBaseType(type:str) -> int: ...
    def IsA(self, type:str) -> int: ...
    @staticmethod
    def IsTypeOf(type:str) -> int: ...
    def NewInstance(self) -> 'vtkSurfaceLICMapper': ...
    def ReleaseGraphicsResources(self, win:'vtkWindow') -> None: ...
    def RenderPiece(self, ren:'vtkRenderer', act:'vtkActor') -> None: ...
    @staticmethod
    def SafeDownCast(o:'vtkObjectBase') -> 'vtkSurfaceLICMapper': ...
    def ShallowCopy(self, __a:'vtkAbstractMapper') -> None: ...

class vtkTextureIO(object):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, __a:'vtkTextureIO') -> None: ...
    @overload
    @staticmethod
    def Write(filename:str, texture:'vtkTextureObject', subset:Sequence[int] =..., origin:Sequence[float]=...) -> None: ...
    @overload
    @staticmethod
    def Write(filename:str, texture:'vtkTextureObject', subset:'vtkPixelExtent', origin:Sequence[float]=...) -> None: ...
