import pydicom as dicom
import os
import numpy as np
from matplotlib import pyplot as plt, cm
import math


def computeArrayRate(errG, dims, dir):
    criteria = np.arange(0.5, 3.5, 0.5)
    rateGlobal = np.zeros(len(criteria), dtype=float)
    #print(np.size(criteria))
    for i in range(0, np.size(criteria)):
        rateGlobal[i] = 100.*np.size(np.where(np.abs(errG) <= criteria[i]))/3./(
            dims[0]*dims[1]*dims[2]-np.size(np.where(np.isnan(errG)))/3.)
        #print(np.size(np.where(np.isnan(errG)))/3.)
    #print(np.size(np.where(np.isnan(errG)))/3.)
    #print(list(rateGlobal))
    #print(list(criteria))
    fig = plt.figure()
    # fig.set_size_inches(4.,3.)
    low = min(rateGlobal)
    high = max(rateGlobal)
    plt.ylim([math.ceil(low-0.5*(high-low)), math.ceil(high+1)])
    plt.ylim([95,101])

    plt.bar(list(criteria), list(rateGlobal), align='center',
            width=0.3, color='yellow', edgecolor='red')
    plt.xlabel('Critère: erreur en %')
    plt.ylabel('Taux de passage en %')
    plt.title('Changement de version Raystation: spots à ' + dir)
    fig.savefig('fig/' + dir + '_Rate_histo.pdf',
                facecolor='w', edgecolor='w', format='pdf')

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
    # startz = int(z/2) - cropz/2
    return img[starty:starty+cropy, :, startx:startx+cropx]

def crop_2D_depth(img, cropy):
    y, z = img.shape
    starty = int(y/2) - int(cropy/2)
    return img[starty:starty+cropy, 0:115]


def draw2DMapDoseTransverse(dim1,dim2,array, dir, namefig,version):
    fig = plt.figure()
    im = plt.pcolormesh(dim1, dim2, array)
    plt.title('Dose à la profondeur du maximum : '+version)
    cbar = plt.colorbar(im)
    cbar.set_label('Dose en Gy')
    #plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel('Y en mm')
    plt.ylabel('X en mm')
    plt.set_cmap(plt.get_cmap('jet'))
    fig.savefig('fig/' + dir + '_DoseMapTransverse' + namefig+version+'.pdf',
                facecolor='w', edgecolor='w', format='pdf')

def draw2DMapDoseDepth(dim1,dim2,array, cropped, dir, namefig,version):
    fig = plt.figure()
    croppedArray = crop_2D_depth(array,cropped)
    im = plt.pcolormesh(dim1, dim2, croppedArray)
    plt.title('Dose en fonctionde la profondeur: '+version)
    cbar = plt.colorbar(im)
    cbar.set_label('Dose en Gy')

   
    #plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel('Z en mm')
    plt.ylabel('X en mm')
    plt.set_cmap(plt.get_cmap('jet'))
    fig.savefig('fig/' + dir + '_DoseMapDepth' + namefig+version+'.pdf',
                facecolor='w', edgecolor='w', format='pdf')

def draw2DMapErrorTransverse(dim1,dim2,array, dir,maxDose):
    fig = plt.figure()
    im = plt.pcolormesh(dim1, dim2, array)
    plt.title('Erreur de dose à la profondeur du maximum')
    cbar = plt.colorbar(im)
    plt.clim(-3, 3)
    cbar.set_label('Erreur en %'+' de '+str(maxDose)+ ' (Gy)')
    #plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel('Y en mm')
    plt.ylabel('X en mm')
    plt.set_cmap(plt.get_cmap('RdYlBu_r'))
    fig.savefig('fig/' + dir + '_ErrorMapTransverse' + '.pdf',
                facecolor='w', edgecolor='w', format='pdf')

def draw2DMapErrorDepth(dim1,dim2,array, cropped, dir,maxDose):
    fig = plt.figure()
    croppedArray = crop_2D_depth(array,cropped)
    im = plt.pcolormesh(dim1, dim2, croppedArray)
    plt.title('Erreur de dose à la profondeur du maximum')
    cbar = plt.colorbar(im)
    plt.clim(-3, 3)
    cbar.set_label('Erreur en %'+' de '+str(maxDose)+ ' (Gy)')
    #plt.axes().set_aspect('equal', 'datalim')
    plt.xlabel('Y en mm')
    plt.ylabel('X en mm')
    plt.set_cmap(plt.get_cmap('RdYlBu_r'))
    fig.savefig('fig/' + dir + '_ErrorMapDepth' +'.pdf',
                facecolor='w', edgecolor='w', format='pdf')



PathDicom = "data/"
#subpath = "MC/CGTR2019/"
#subpath = "MC/CGTR2017/"
#subpath = "PB/CGTR2019/"
subpath = "PB/CGTR2017/"
PathDicom = PathDicom +subpath
refFile = "RS8-b.dcm"
testedFile = "RS10-a.dcm"
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
        #print(refDs)
        
        ds = refDs.pixel_array
        doseScalingRef = refDs.DoseGridScaling
        doseScalingTested =testedDs.DoseGridScaling
        #print(doseScalingRef)
        #print(doseScalingTested)
        ConstPixelDims = ds.shape

        # print(ConstPixelDims)
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

        cropped = 100
        lx = crop_1D(x, cropped)
        ly = crop_1D(y, cropped)
        lz = z[0:115]
        # print(len(x),len(y),len(z))
        # print(lx)

        # The array is sized based on 'ConstPixelDims'
        refArray = np.zeros(ConstPixelDims, dtype=refDs.pixel_array.dtype)
        testedArray = np.zeros(
            ConstPixelDims, dtype=testedDs.pixel_array.dtype)
        # print(np.shape(refDs.pixel_array))
        # print(np.shape(refArray))
        # print(len(refArray[0]),len(refArray[1]),len(refArray[2]))
        refArray[:, :, :] = refDs.pixel_array#*doseScalingRef
        testedArray[:, :, :] = testedDs.pixel_array#*doseScalingTested
        refArray = refArray*doseScalingRef
        BraggPeakRef = np.sum(np.sum(refArray,axis=2),axis=0)
        #print(BraggPeakRef)
        zmax=np.argmax(BraggPeakRef)
        testedArray = testedArray*doseScalingTested
        maxA = np.max(refArray)
        errorGlobal = 100*(refArray-testedArray)/maxA
        #print(maxA)
        errorGlobal[refArray == 0.] = np.nan # where it's 0 data do not considered for statistics 
        d=subpath+d
        computeArrayRate(errorGlobal, ConstPixelDims, d)
        #print(np.size(np.where(refArray == 0.))/3.)
        #print(len(refArray))
        lrefArray = crop_3D(refArray, cropped, cropped)
        ltestedArray = crop_3D(testedArray, cropped, cropped)
        lerrorGlobal = crop_3D(errorGlobal, cropped, cropped)

        MaxIndices = np.where(refArray == np.amax(refArray))
        ref = 'RS8'
        tested = 'RS10'
        #print(MaxIndices)
        if MaxIndices[0][0] != 85:
            MaxIndices[0][0] =85
        
        draw2DMapDoseTransverse(ly, lx, lrefArray[:, zmax, :],d, 'AtMaxDepth',ref)
        draw2DMapDoseTransverse(ly, lx, ltestedArray[:, zmax, :],d, 'AtMaxDepth',tested)
        draw2DMapErrorTransverse(ly, lx, lerrorGlobal[:, zmax, :],d,np.round(maxA,2))

        draw2DMapDoseDepth(lz, lx, refArray[:, :, MaxIndices[0][0]], cropped, d, 'AtMaxDepth',ref)
        draw2DMapDoseDepth(lz, lx, testedArray[:, :, MaxIndices[0][0]], cropped, d, 'AtMaxDepth',tested)
        draw2DMapErrorDepth(lz, lx, errorGlobal[:,:,MaxIndices[0][0]],cropped,d,np.round(maxA,2))
