import pydicom as dicom
import os
import numpy as np
from matplotlib import pyplot as plt, cm


def crop_1D(img, cropx):
    x = img.shape
    # print(x[0])
    startx = int(x[0]/2) - int(cropx/2)
    # print(startx)
    return img[startx:startx+cropx]


def crop_3D(img, cropx, cropy):
    y, x, z = img.shape
    startx = int(x/2) - int(cropx/2)
    starty = int(y/2) - int(cropy/2)
    #startz = int(z/2) - cropz/2
    return img[starty:starty+cropy, :, startx:startx+cropx]


PathDicom = "data/"
refFile = "RS6.dcm"
testedFile = "RS8.dcm"
# lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for d in subdirList:
        rFile = os.path.join(dirName+d, refFile)
        tFile = os.path.join(dirName+d, testedFile)
        print(rFile, tFile)
        # for filename in fileList:
        #    if ".dcm" in filename.lower():  # check whether the file's DICOM
        #        lstFilesDCM.append(os.path.join(dirName,filename))

        refDs = dicom.read_file(rFile)
        testedDs = dicom.read_file(tFile)
        ds = refDs.pixel_array

        # (int(refDs.Lines),int(refDs.Rows), int(refDs.Columns))
        ConstPixelDims = ds.shape
        print(ConstPixelDims)

        # Load spacing values (in mm)
        ConstPixelSpacing = (float(refDs.PixelSpacing[0]), float(
            refDs.PixelSpacing[1]), float(refDs.SliceThickness))
        # print(ConstPixelSpacing)

        x = np.arange(
            0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
        y = np.arange(
            0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
        z = np.arange(
            0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
        #x = x[int(len(x)/2)-50:int(len(x)/2)+50]
        cropped = 80
        lx = crop_1D(x, cropped)
        ly = crop_1D(y, cropped)
        
        # print(len(x),len(y),len(z))
        # print(lx)
        # The array is sized based on 'ConstPixelDims'
        refArray = np.zeros(ConstPixelDims, dtype=refDs.pixel_array.dtype)
        testedArray = np.zeros(ConstPixelDims, dtype=testedDs.pixel_array.dtype)
        # print(np.shape(refDs.pixel_array))
        # print(np.shape(refArray))
        # print(len(refArray[0]),len(refArray[1]),len(refArray[2]))
        refArray[:, :, :] = refDs.pixel_array
        testedArray[:, :, :] = testedDs.pixel_array
        errorLocal = 100*(refArray-testedArray)/refArray
        errorGlobal = 100*(refArray-testedArray)/np.max(refArray)
        lrefArray = crop_3D(refArray,cropped,cropped)
        ltestedArray = crop_3D(testedArray,cropped,cropped)
        MaxIndices = np.where(refArray == np.amax(refArray))
        print(MaxIndices)
        plt.figure(dpi=300)
        #plt.axes().set_aspect('equal', 'datalim')
        plt.set_cmap(plt.jet())
        plt.pcolormesh(y, z, refArray[80, :, :])
        plt.pcolormesh(y, x, refArray[:, :, 80])
        plt.pcolormesh(y, x, refArray[:, 80, :])
        plt.pcolormesh(y, x, refArray[:, MaxIndices[2][0], :])
        plt.pcolormesh(y, x, errorGlobal[:, MaxIndices[2][0], :])
        #plt.pcolormesh(ly, lx,lrefArray[:, MaxIndices[2][0], :])
        #print(refArray[:, :, 80])
        rateG=100.*np.size(np.where(errorGlobal <=1))/3./(ConstPixelDims[0]*ConstPixelDims[1]*ConstPixelDims[2]-np.size(np.where(np.isnan(errorGlobal)))/3.)
        print(rateG)
        rateL=100.*np.size(np.where(errorLocal <=100))/3./(ConstPixelDims[0]*ConstPixelDims[1]*ConstPixelDims[2]-np.size(np.where(np.isnan(errorLocal)))/3.)
        print(rateL)
        #plt.show()
