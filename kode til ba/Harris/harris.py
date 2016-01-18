from __future__ import division
import cv2
import numpy
import math
import scipy
from scipy import ndimage
from scipy import misc
from scipy import linalg as LA
import helperfunctions as h

# 3 6400 = 1372
# 1 200000000 =  1393 
# 2 8400 = 1372

def harris(Iname, k, thresh,flag):
  """
  Input : An image, value k and threshold
  Output: Finds interest points using -
          the feature detecor method "Harris corner detector"
  """
  if flag == 1:
    method = "Harris"
    print "Using Harris method with a threshold of:", thresh, "recommended: 200000000"
  if flag == 2:
    method = "HoShiThomas"
    print "Using HoShiThomas method with a threshold of:", thresh, "recommended: 8400"
  if flag == 3:
    method = "Noble"
    print "Using noble method with a threshold of:", thresh, "recommended: 6400"
  # Create empty image holders:
  I = cv2.imread(Iname)
  I0 = ndimage.filters.gaussian_filter(cv2.imread(Iname, 0).astype(float), 4.52)
  I0 = misc.imresize(I0, 25, 'bilinear').astype(float)
  C = numpy.empty([len(I0),len(I0[0])]).astype(float)

  # Convolution masks for derivatives and smoothing:
  mask_dx = numpy.array([[-1,0,1],[-1,0,1],[-1,0,1]]).astype(float)
  mask_dy = numpy.transpose(mask_dx)

  Ix = cv2.filter2D(I0, -1, mask_dx) # x derivative
  Iy = cv2.filter2D(I0, -1, mask_dy) # y derivative
  sigma = 1.5
  Ixy = ndimage.filters.gaussian_filter(Ix * Iy, sigma)
  Ixx = ndimage.filters.gaussian_filter(Ix ** 2, sigma)
  Iyy = ndimage.filters.gaussian_filter(Iy ** 2, sigma)
  Ixx = Ix ** 2
  Iyy = Iy ** 2

  """
  cv2.imshow("Img1", Ixy)
  cv2.waitKey(0)
  cv2.imshow("Img2", Ixx)
  cv2.waitKey(0)
  cv2.imshow("Img3", Iyy)
  cv2.waitKey(0)
  """

  # Accordingly to the notation in Harris' paper:
  # A = Ixx
  # B = Iyy
  # C = Ixy

  harrisExtremaPoints = []
  hoShiThomasExtremaPoints = []
  nobleExtremaPoints = []
  
  # Calculate R = Det -k * TR^2:
  for y in range(0, len(I0) ):
    for x in range(0, len(I0[0])):
      a = Ixx[y, x]
      b = Iyy[y, x]
      c = Ixy[y, x]
      # Harris method:
      if flag == 1:
        C[y, x] = (a * b - c**2) - k * (a + b)**2
      # HoShiThomas method:  
      if flag == 2:
        M = numpy.array([[a, c],[c, b]])
        e_vals, e_vecs = LA.eig(M)
        e_vals = e_vals.real
        C[y][x] = min(e_vals)
      # Noble method:
      if flag == 3:
        epsilon = 0.2
        M = numpy.array([[a, c],[c, b]])
        det = (a*b)-(c*c)
        trace = a + b
        C[y][x] = det/(trace+epsilon)      
  
  # Threshold values to perform edge hysteresis:
  for y in range(3, len(C)):
    for x in range(3, len(C[0])):
      if flag == 1:
        if (C[y, x] > thresh):
          harrisExtremaPoints.append([y, x, 11])
          I[y, x] = [0,0,255]
      if flag == 2:
        if (C[y][x] > thresh):
          hoShiThomasExtremaPoints.append([y,x])
          I[y][x] = [0,0,255]
      if flag == 3:
        if (C[y][x] > thresh):
          nobleExtremaPoints.append([y,x])
          I[y][x] = [0,0,255]

  numpy.save(Iname[:-4], numpy.array(harrisExtremaPoints))
  if flag == 1:
    print "Found\t",len(harrisExtremaPoints)
    print "Name\t" + Iname
    return(harrisExtremaPoints)
  if flag == 2:
    h.points_to_txt(hoShiThomasExtremaPoints, "hoShiThomasExtremaPoints.txt", "\n")
    print "Found",len(hoShiThomasExtremaPoints), "interest points. threshold:",thresh
    return(hoShiThomasExtremaPoints)
  if flag == 3:
    h.points_to_txt(nobleExtremaPoints, "nobleExtremaPoints.txt", "\n")
    print "Found",len(nobleExtremaPoints), "interest points. threshold:",thresh
    return(nobleExtremaPoints)

def test_corner_methods(filename,thresh1,thresh2,thresh3):
  harris(filename,0.04,thresh1,1)
  harris(filename,0,thresh2,2)
  harris(filename,0,thresh3,3)

#test_corner_methods('erimitage2.jpg', 200000000, 8400, 6400)
harris('IMG_9370.jpg', 0.04, 100000000, 1)
