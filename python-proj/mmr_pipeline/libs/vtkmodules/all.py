""" This module loads the entire VTK library into its namespace.  It
also allows one to use specific packages inside the vtk directory.."""

from __future__ import absolute_import

# --------------------------------------
from .vtkCommonCore import *
from .vtkWebCore import *
from .vtkCommonMath import *
from .vtkCommonTransforms import *
from .vtkCommonDataModel import *
from .vtkCommonExecutionModel import *
from .vtkIOCore import *
from .vtkImagingCore import *
from .vtkIOImage import *
from .vtkIOXMLParser import *
from .vtkIOXML import *
from .vtkCommonMisc import *
from .vtkFiltersCore import *
from .vtkRenderingCore import *
from .vtkRenderingContext2D import *
from .vtkRenderingFreeType import *
from .vtkRenderingSceneGraph import *
from .vtkRenderingVtkJS import *
from .vtkIOExport import *
from .vtkWebGLExporter import *
from .vtkCommonComputationalGeometry import *
from .vtkCommonSystem import *
from .vtkIOLegacy import *
from .vtkDomainsChemistry import *
from .vtkFiltersSources import *
from .vtkFiltersGeneral import *
from .vtkRenderingHyperTreeGrid import *
from .vtkRenderingUI import *
from .vtkRenderingOpenGL2 import *
from .vtkRenderingContextOpenGL2 import *
from .vtkRenderingVolume import *
from .vtkImagingMath import *
from .vtkRenderingVolumeOpenGL2 import *
from .vtkInteractionWidgets import *
from .vtkViewsCore import *
from .vtkViewsContext2D import *
from .vtkTestingRendering import *
from .vtkInteractionStyle import *
from .vtkViewsInfovis import *
from .vtkRenderingVolumeAMR import *
from .vtkPythonContext2D import *
from .vtkRenderingParallel import *
from .vtkRenderingVR import *
from .vtkRenderingMatplotlib import *
from .vtkRenderingLabel import *
from .vtkRenderingLOD import *
from .vtkRenderingLICOpenGL2 import *
from .vtkRenderingImage import *
from .vtkRenderingExternal import *
from .vtkFiltersCellGrid import *
from .vtkRenderingCellGrid import *
from .vtkIOXdmf2 import *
from .vtkIOVeraOut import *
from .vtkIOVPIC import *
from .vtkIOTecplotTable import *
from .vtkIOTRUCHAS import *
from .vtkIOSegY import *
from .vtkIOParallelXML import *
from .vtkIOLSDyna import *
from .vtkIOParallelLSDyna import *
from .vtkIOExodus import *
from .vtkIOParallelExodus import *
from .vtkIOPLY import *
from .vtkIOPIO import *
from .vtkIOMovie import *
from .vtkIOOggTheora import *
from .vtkIOOMF import *
from .vtkIONetCDF import *
from .vtkIOMotionFX import *
from .vtkIOGeometry import *
from .vtkIOParallel import *
from .vtkIOMINC import *
from .vtkIOInfovis import *
from .vtkIOImport import *
from .vtkParallelCore import *
from .vtkIOIOSS import *
from .vtkIOH5part import *
from .vtkIOH5Rage import *
from .vtkIOGeoJSON import *
from .vtkIOFLUENTCFF import *
from .vtkIOVideo import *
from .vtkIOExportPDF import *
from .vtkRenderingGL2PSOpenGL2 import *
from .vtkIOExportGL2PS import *
from .vtkIOEnSight import *
from .vtkIOCityGML import *
from .vtkIOChemistry import *
from .vtkIOCesium3DTiles import *
from .vtkIOCellGrid import *
from .vtkIOCONVERGECFD import *
from .vtkIOHDF import *
from .vtkIOCGNSReader import *
from .vtkIOAsynchronous import *
from .vtkIOAMR import *
from .vtkInteractionImage import *
from .vtkImagingStencil import *
from .vtkImagingStatistics import *
from .vtkImagingGeneral import *
from .vtkImagingOpenGL2 import *
from .vtkImagingMorphological import *
from .vtkImagingFourier import *
from .vtkIOSQL import *
from .vtkCommonColor import *
from .vtkImagingSources import *
from .vtkInfovisCore import *
from .vtkGeovisCore import *
from .vtkInfovisLayout import *
from .vtkRenderingAnnotation import *
from .vtkImagingHybrid import *
from .vtkImagingColor import *
from .vtkFiltersTopology import *
from .vtkFiltersTensor import *
from .vtkFiltersSelection import *
from .vtkFiltersSMP import *
from .vtkFiltersReduction import *
from .vtkFiltersPython import *
from .vtkFiltersProgrammable import *
from .vtkFiltersModeling import *
from .vtkFiltersPoints import *
from .vtkFiltersStatistics import *
from .vtkFiltersParallelStatistics import *
from .vtkFiltersImaging import *
from .vtkFiltersExtraction import *
from .vtkFiltersGeometry import *
from .vtkFiltersHybrid import *
from .vtkFiltersHyperTree import *
from .vtkFiltersTexture import *
from .vtkFiltersParallel import *
from .vtkFiltersParallelImaging import *
from .vtkFiltersParallelDIY2 import *
from .vtkFiltersGeometryPreview import *
from .vtkFiltersGeneric import *
from .vtkFiltersFlowPaths import *
from .vtkFiltersAMR import *
from .vtkDomainsChemistryOpenGL2 import *
from .vtkCommonPython import *
from .vtkChartsCore import *
from .vtkAcceleratorsVTKmCore import *
from .vtkAcceleratorsVTKmDataModel import *
from .vtkAcceleratorsVTKmFilters import *
from .vtkFiltersVerdict import *


# useful macro for getting type names
from .util.vtkConstants import vtkImageScalarTypeNameMacro

# import convenience decorators
from .util.misc import calldata_type

# import the vtkVariant helpers
from .util.vtkVariant import *