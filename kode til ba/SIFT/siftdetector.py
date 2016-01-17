from __future__ import division
import scipy
import numpy as np
from scipy import ndimage
from scipy import misc
import cv2
import math
import csv
from PIL import Image
import helperfunctions as h
import siftorientation as ss

def find_max_new(dog_scale,i,y,x,princip_cur, xht):
  maxpoint = (dog_scale[i, y, x] > 0)
  minpoint = (dog_scale[i, y, x] < 0)
  # Run through 26 neighbours
  for ci in range(-1,2):
    for cy in range(-1,2):
      for cx in range(-1,2):
        if cy == 0 and cx == 0 and ci == 0:
          continue # perform next iteration as we are in orego.
        maxpoint = maxpoint and dog_scale[i,y,x]>dog_scale[i+ci,y+cy,x+cx]
        minpoint = minpoint and dog_scale[i,y,x]<dog_scale[i+ci,y+cy,x+cx]
        # If point lies between max and min, we break
        if not maxpoint and not minpoint:
          return 0
      if not maxpoint and not minpoint:
        return 0
    if not maxpoint and not minpoint:
      return 0
  if maxpoint == True or minpoint == True:
    # Create array of neighbouring points 
    Dxx = (dog_scale[i,y,x+1] + dog_scale[i,y,x-1] - 2 * dog_scale[i,y,x]) * 1.0 / 255
    Dyy = (dog_scale[i,y+1,x] + dog_scale[i,y-1,x] - 2 * dog_scale[i,y,x]) * 1.0 / 255
    Dss = (dog_scale[i+1,y,x] + dog_scale[i-1,y,x] - 2 * dog_scale[i,y,x]) * 1.0 / 255
    Dxy = (dog_scale[i,y+1,x+1] - dog_scale[i,y+1,x-1] - dog_scale[i,y-1,x+1] + dog_scale[i,y-1,x-1]) * 0.25 / 255
    Dxs = (dog_scale[i+1,y,x+1] - dog_scale[i+1,y,x-1] - dog_scale[i-1,y,x+1] + dog_scale[i-1,y,x-1]) * 0.25 / 255 
    Dys = (dog_scale[i+1,y+1,x] - dog_scale[i+1,y-1,x] - dog_scale[i-1,y+1,x] + dog_scale[i-1,y-1,x]) * 0.25 / 255  
    #3x3 dimensional Hessian
    H = np.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
    det = float(np.linalg.det(H))

    DXX1 = Dxx * 255
    DYY1 = Dyy * 255
    DSS1 = Dss * 255
    DXY1 = Dxy * 255
    DXS1 = Dxs * 255
    DYS1 = Dys * 255
    #2x2 dimensional Hessian
    H2 = np.matrix([[Dxx, Dxy], [Dxy, Dyy]]) 
    det2 = float(np.linalg.det(H2))

    if (det != 0):
      Dx = (dog_scale[i,y,x+1] - dog_scale[i,y,x-1]) * 0.5 / 255
      Dy = (dog_scale[i,y+1,x] - dog_scale[i,y-1,x]) * 0.5 / 255
      Ds = (dog_scale[i+1,y,x] - dog_scale[i-1,y,x]) * 0.5 / 255
      DX = np.matrix([[Dx], [Dy], [Ds]])
      tr = Dxx + Dyy #float(Dxx) + float(DYY1)
      r = float(princip_cur)
      xhat = np.linalg.inv(H) * DX
      if (abs(xhat[0]) < xht and abs(xhat[1]) < xht and abs(xhat[2]) < xht and det2 > 0):
        Dxhat = dog_scale[i,y,x] + (1/2.0) * DX.transpose() * xhat # CT old was point, but shouldnt differ
        if((abs(Dxhat) > 1.03) and (tr**2/det2 < (r + 1)**2 / r)):
          return 1
    return 0

def SIFT(filename, r_mag):
  """
  Returns the interest points found
  """
  k_con = math.sqrt(2)
  k = k_con
  """
  sigma1 = np.array([k_con/2, 1.0, k_con , k_con ** 2, k_con ** 3])
  sigma2 = np.array([k_con, k_con ** 2, k_con ** 3, k_con ** 4, k_con ** 5])
  sigma3 = np.array([k_con ** 3, k_con ** 4, k_con ** 5, k_con ** 6, k_con ** 7])
  sigma4 = np.array([k_con ** 5, k_con ** 6, k_con ** 7, k_con ** 8, k_con ** 9])
  sigma5 = np.array([k_con ** 10, k_con ** 11, k_con ** 12, k_con ** 13, k_con ** 14])
  """

  sigma0 = np.array([1.3, 1.6, 1.6 * k, 1.6 * k ** 2, 1.6 * k ** 3, \
                    1.6 * k, 1.6 * k ** 2, 1.6 * k ** 3, 1.6 * k ** 4, 1.6 * k ** 5, \
                    1.6 * k ** 3, 1.6 * k ** 4, 1.6 * k ** 5, 1.6 * k ** 6, 1.6 * k ** 7, \
                    1.6 * k ** 5, 1.6 * k ** 6, 1.6 * k ** 7, 1.6 * k ** 8, 1.6 * k ** 9, \
                    1.6 * k ** 7, 1.6 * k ** 8, 1.6 * k ** 9, 1.6 * k ** 10, 1.6 * k ** 11])


  I = cv2.imread(filename).astype(float)
  I0 = cv2.imread(filename, 0).astype(float)
  I1 = misc.imresize(I0, 1/2.0, 'bilinear').astype(float)
  I2 = misc.imresize(I0, 1/4.0, 'bilinear').astype(float)
  I3 = misc.imresize(I0, 1/8.0, 'bilinear').astype(float)
  I4 = misc.imresize(I0, 1/16.0, 'bilinear').astype(float)

  print "creating gaussian pyramids.."
  o0sc = np.zeros((5, I0.shape[0],I0.shape[1]))
  o1sc = np.zeros((5, I1.shape[0],I1.shape[1]))
  o2sc = np.zeros((5, I2.shape[0],I2.shape[1]))
  o3sc = np.zeros((5, I3.shape[0],I3.shape[1]))
  o4sc = np.zeros((5, I4.shape[0],I4.shape[1]))

  for i in range(0,5):
    """
    o1sc[i,:,:] = cv2.filter2D(I_bw, -1, h.gauss(9, sigma1[i]))
    o2sc[i,:,:] = misc.imresize(cv2.filter2D(I_bw, -1, h.gauss(9,sigma2[i])), 50, 'bilinear').astype(float)
    o3sc[i,:,:] = misc.imresize(cv2.filter2D(I_bw, -1, h.gauss(9,sigma3[i])), 25, 'bilinear').astype(float)
    o4sc[i,:,:] = misc.imresize(cv2.filter2D(I_bw, -1, h.gauss(9,sigma4[i])), 1.0/8.0, 'bilinear').astype(float)
    """
    o0sc[i,:,:] = ndimage.filters.gaussian_filter(I0, sigma0[i])
    o1sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma0[i + 5]), 1/2.0, 'bilinear')
    o2sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma0[i + 10]), 1/4.0, 'bilinear')
    o3sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma0[i + 15]), 1/8.0, 'bilinear')
    o4sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma0[i + 20]), 1/16.0, 'bilinear')
    """
    cv2.imwrite("o1sc-sci" + str(filename[:-4]) + str(i) + ".jpg", o1sc[i])
    cv2.imwrite("o2sc-sci" + str(filename[:-4]) + str(i) + ".jpg", o2sc[i])
    cv2.imwrite("o3sc-sci" + str(filename[:-4]) + str(i) + ".jpg", o3sc[i])
    cv2.imwrite("o4sc-sci" + str(filename[:-4]) + str(i) + ".jpg", o4sc[i])
    """

  # Calculate difference of gaussian images.
  print "creating difference of gaussian pyramids.."
  DoG_pictures_scale_0 = np.zeros((4, I0.shape[0],I0.shape[1]))
  DoG_pictures_scale_1 = np.zeros((4, I1.shape[0],I1.shape[1]))
  DoG_pictures_scale_2 = np.zeros((4, I2.shape[0],I2.shape[1]))
  DoG_pictures_scale_3 = np.zeros((4, I3.shape[0],I3.shape[1]))
  DoG_pictures_scale_4 = np.zeros((4, I4.shape[0],I4.shape[1]))
  #print(DoG_pictures_scale_4)

  for i in range(0,4):
    # CT: TRY WITH HELPERFUNCTION MINUS
    DoG_pictures_scale_0[i,:,:] = o0sc[i + 1,:,:] - o0sc[i,:,:]
    DoG_pictures_scale_1[i,:,:] = o1sc[i + 1,:,:] - o1sc[i,:,:]
    DoG_pictures_scale_2[i,:,:] = o2sc[i + 1,:,:] - o2sc[i,:,:]
    DoG_pictures_scale_3[i,:,:] = o3sc[i + 1,:,:] - o3sc[i,:,:]
    DoG_pictures_scale_4[i,:,:] = o4sc[i + 1,:,:] - o4sc[i,:,:]

  #print DoG_pictures_scale_1[:,:,1]
  #cv2.imshow('image',DoG_pictures_scale_1[:,:,0])
  #cv2.waitKey(0)
  
  # Initialize arrays for keypoints
  DoG_extrema_points_0_1 = []
  DoG_extrema_points_0_2 = []
  DoG_extrema_points_1_1 = []
  DoG_extrema_points_1_2 = []
  DoG_extrema_points_2_1 = []
  DoG_extrema_points_2_2 = []
  DoG_extrema_points_3_1 = []
  DoG_extrema_points_3_2 = []
  DoG_extrema_points_4_1 = []
  DoG_extrema_points_4_2 = []
   
  """
  print("Finding points for octave 1")
  for y in range(3, I1.shape[0] - 3):
    for x in range(3, I1.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_1,1,y,x, 1, 0.5) == 1):
        DoG_extrema_points_1_1.append([y,x,6])
        cv2.circle(I, (x * 2, y * 2), 6, (0,0,255), 2)
      if (find_max_new(DoG_pictures_scale_1,2,y,x, 1, 0.5) == 1):
        DoG_extrema_points_1_2.append([y,x, 7])
        cv2.circle(I, (x * 2, y * 2), 6, (0,0,125), 2)
  dogn1 = np.array(DoG_extrema_points_1_1)
  dogn2 = np.array(DoG_extrema_points_1_2)
  result1 = np.vstack([dogn1, dogn2])
  print "Number of points in first octave: %d" % len(result1)
  """

  print "Finding points for octave 2"
  for y in range(3, I2.shape[0] - 3):
    for x in range(3,  I2.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_2,1,y,x, 3, 0.5) == 1):
        DoG_extrema_points_2_1.append([y,x, 11])
        cv2.circle(I, (x * 4, y * 4), 12, (0,255,0), 2)
      if (find_max_new(DoG_pictures_scale_2,2,y,x, 3, 0.5) == 1):
        DoG_extrema_points_2_2.append([y,x, 12])
        cv2.circle(I, (x * 4, y * 4), 12, (0,125,0), 2)
  dogn1 =  np.array(DoG_extrema_points_2_1)
  dogn2 =  np.array(DoG_extrema_points_2_2)
  print(len(dogn1), len(dogn2))
  result2 = np.vstack([dogn1, dogn2])
  print "Number of points in second octave: %d" % len(result2)

  """
  print "Finding points for octave 3"
  for y in range(3, I3.shape[0] - 3):
    for x in range(3,  I3.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_3,1,y,x, 15, 2.5) == 1):
        DoG_extrema_points_3_1.append([y,x, 16])
        cv2.circle(I, (x * 8, y * 8), 24, (255,0,0), 2)
      if (find_max_new(DoG_pictures_scale_3,2,y,x, 15, 2.5) == 1):
        DoG_extrema_points_3_2.append([y,x, 17])
        cv2.circle(I, (x * 8, y * 8), 24, (125, 0,0), 2)
  dogn1 =  np.array(DoG_extrema_points_3_1)
  dogn2 =  np.array(DoG_extrema_points_3_2)
  print(len(dogn1), len(dogn2))
  result3 = np.vstack([dogn1, dogn2])
  #print "Number of points in third octave: %d" % len(result3)

  print "Finding points for octave 4"
  for y in range(3, I4.shape[0] - 3):
    for x in range(3,  I4.shape[1] - 3):
      if (find_max_new(DoG_pictures_scale_4,1,y,x, 15, 1.5) == 1):
        DoG_extrema_points_4_1.append([y,x, 21])
        cv2.circle(I, (x * 16, y * 16), 30, (125,125,0), 2)
      if (find_max_new(DoG_pictures_scale_4,2,y,x, 15, 1.5) == 1):
        DoG_extrema_points_4_2.append([y,x, 22])
        cv2.circle(I, (x * 16, y * 16), 30, (125, 200,0), 2)
  dogn1 =  np.array(DoG_extrema_points_4_1)
  dogn2 =  np.array(DoG_extrema_points_4_2)
  print(len(dogn1), len(dogn2))
  result4 = np.vstack([dogn1, dogn2])
  #print "Number of points in fourth octave: %d" % len(result4)
  cv2.imwrite("xx-des-" + str(filename[:-4]) + ".jpg", I)
  """
  ret = np.vstack([result2])

  return(ret)
