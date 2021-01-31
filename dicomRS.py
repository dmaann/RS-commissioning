import pydicom as dicom
import os
import numpy
from matplotlib import pyplot, cm

PathDicom = "data/test/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])
# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ds = RefDs.pixel_array  

ConstPixelDims = ds.shape #(int(RefDs.Lines),int(RefDs.Rows), int(RefDs.Columns))
print(ConstPixelDims)

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
#print(ConstPixelSpacing)

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

print(len(x),len(y),len(z))
print(x)
# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
print(numpy.shape(RefDs.pixel_array))
print(numpy.shape(ArrayDicom))
#print(len(ArrayDicom[0]),len(ArrayDicom[1]),len(ArrayDicom[2]))
ArrayDicom[:,:,:] = RefDs.pixel_array 
#print(ArrayDicom)
# loop through all the DICOM files
'''for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    print(len(ds.pixel_array[0]),len(ds.pixel_array[1]),len(ds.pixel_array[2]))
    # store the raw image data
    #print(ds.pixel_array)
    ArrayDicom[:, :,lstFilesDCM.index(filenameDCM)] = ds.pixel_array  
'''
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.viridis())
pyplot.pcolormesh(y, z, numpy.flipud(ArrayDicom[80, :, :]))
pyplot.pcolormesh(x, z, numpy.flipud(ArrayDicom[:, 80, :]))
print(ArrayDicom[:, :, 80])
pyplot.show()