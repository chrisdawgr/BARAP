from __future__ import print_function
from __future__ import division
import scipy
import numpy
import scipy.ndimage
from scipy import misc
import cv2
import math
import csv
from PIL import Image
import helperfunctions as h
from scipy import signal
from scipy import misc
from scipy import ndimage

# Dedicated to my dear, my muse, my Andreas

def find_max_new(scale,i,y,x):
  """
  Input : scale of point, i, y, x
  Output: max of 3x3x3 area
  """
  maxpoint = (scale[i, y, x] > 0)
  minpoint = (scale[i, y, x] < 0)
  # Run through 26 neighbours
  for ci in range(-1,2):
    for cy in range(-1,2):
      for cx in range(-1,2):
        if cy == 0 and cx == 0 and ci == 0:
          continue # perform next iteration as we are in orego.
        maxpoint = maxpoint and scale[i,y,x]>scale[i+ci,y+cy,x+cx]
        minpoint = minpoint and scale[i,y,x]<scale[i+ci,y+cy,x+cx]
        #print scale[y+cy,x+cx,i+ci]
        # If point lies between max and min, we break
        if not maxpoint and not minpoint:
          return 0
      if not maxpoint and not minpoint:
        return 0
    if not maxpoint and not minpoint:
      return 0
  if maxpoint == True or minpoint == True:
    return 1

def accurate_keypoint(point,deriv1, deriv2, deriv3, xh_t):
  """
  Input : point(y,x), derivates d1 d2 d3, a limit for xhat 
  Output: returns 1 if the point is at an extreme
  """
  # deriv [dxx,dyy,dxy,dx,dy,gauss]
  Dxx = deriv2[0] * 1.0
  Dyy = deriv2[1] * 1.0
  Dxy = deriv2[2] * 0.25
  Dxs = (deriv3[3]-deriv1[3])* 0.25
  Dys = (deriv3[4]-deriv1[4])* 0.25
  Dss = (-2 * deriv2[5] + deriv3[5] + deriv1[5]) * 1.0
  Dx = deriv2[3] * 0.5
  Dy = deriv2[4] * 0.5
  Ds = (deriv3[5] - deriv1[5]) * 0.5 
  H = numpy.matrix([[Dxx, Dxy, Dxs], [Dxy, Dyy, Dys], [Dxs, Dys, Dss]])
  det = float(numpy.linalg.det(H))
  DX = numpy.matrix([[Dx], [Dy], [Ds]])
  if det != 0:
    xhat = numpy.linalg.inv(H) * DX
    #print(xhat)
    #print("\n")
    if (abs(xhat[0]) < xh_t and abs(xhat[1]) < xh_t and abs(xhat[2]) < xh_t):
      Dxhat = point + (1/2.0) * DX.transpose() * xhat #  This is way too big. Missing point
      #print(Dxhat)
      #print("\n")
      if(abs(Dxhat) > 0.5):
        return 1
      #print ("rejected dxhat")
    #print ("rejected xhat")
    return 0
  return 0

def getGauss(size,i):
  """
  Input : size of filter, sigma
  Output: retrns filter
  """
  sigma = 1.2
  if i == 0:
    return h.gauss2x(size,sigma)
  if i == 1:
    return h.gauss2y(size,sigma)
  if i == 2:
    return h.gauss2xy(size,sigma)
  if i == 3:
    return h.gaussdx(size,sigma)
  if i == 4:
    return h.gaussdy(size,sigma)
  if i == 5:
    return h.gauss(size,sigma)

def findSurfPoints(filename):
  """
  Input : name of file
  Output: interest points
  """
  clear = " " * 50
  I_bw = cv2.imread(filename, 0).astype(float)/255.0
  I = cv2.imread(filename)
  
  # Initialize gaussian kernel holders
  filter9 =  numpy.zeros((6,9,9))
  filter15 = numpy.zeros((6,15,15))
  filter21 = numpy.zeros((6,21,21))
  filter27 = numpy.zeros((6,27,27))
  filter39 = numpy.zeros((6,39,39))
  filter51 = numpy.zeros((6,51,51))
  filter75 = numpy.zeros((6,75,75))
  filter99 = numpy.zeros((6,99,99))

  #print("Process: Calculating Gaussian kernels","\r", end="")
  # Get gaussian kernels [dxx,dyy,dxy,dx,dy,gauss]
  #numpy.set_printoptions(precision=0)
  for i in range(0,6):
    filter9[i] =  getGauss(9,i)
    filter15[i] = getGauss(15,i)
    filter21[i] = getGauss(21,i)
    filter27[i] = getGauss(27,i)
    filter39[i] = getGauss(39,i)
    filter75[i] = getGauss(75,i)
    filter99[i] = getGauss(99,i)
  #print ("\n")
  #print (numpy.sum(filter9[:,:,1]))
  #print (numpy.sum(filter9[:,:,0]))

  # Intitialize convolved image holder
  conv9  = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv15 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv21 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv27 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv39 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv51 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv75 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))
  conv99 = numpy.zeros((6, I_bw.shape[0],I_bw.shape[1]))

  # Each holder has the image corresponding to dxx, dyy, dxy, dx, dy, gauss
  #print(clear,"\r", end="")
  #print("Process: Convolving image for all derivates","\r", end="")
  for i in range(0,6):
    conv9[i]  = cv2.filter2D(I_bw,-1, filter9[i])
    conv15[i] = cv2.filter2D(I_bw,-1, filter15[i])
    conv21[i] = cv2.filter2D(I_bw,-1, filter21[i])
    conv27[i] = cv2.filter2D(I_bw,-1, filter27[i])
    conv39[i] = cv2.filter2D(I_bw,-1, filter39[i])
    conv51[i] = cv2.filter2D(I_bw,-1, filter51[i])
    conv75[i] = cv2.filter2D(I_bw,-1, filter75[i])
    conv99[i] = cv2.filter2D(I_bw,-1, filter99[i])
    #cv2.imshow('image', conv9[i])
    #cv2.waitKey(0)
  #print (conv9[:,:,2])
  
  # Initialize holders for determinants of hessian
  o1 = numpy.zeros((4, I_bw.shape[0],I_bw.shape[1]))
  o2 = numpy.zeros((4, I_bw.shape[0],I_bw.shape[1]))
  o3 = numpy.zeros((4, I_bw.shape[0],I_bw.shape[1]))

  # Calculate determinant for each octave
  for y in range(3,I_bw.shape[0]-3):
    for x in range(3,I_bw.shape[1]-3):
      o1[0, y,x] =  conv9[0,y,x]* conv9[1,y,x]-((conv9[2,y,x])**2)
      o1[1, y,x] = conv15[0,y,x]*conv15[1,y,x]-((conv15[2,y,x])**2)
      o1[2, y,x] = conv21[0,y,x]*conv21[1,y,x]-((conv21[2,y,x])**2)
      o1[3, y,x] = conv27[0,y,x]*conv27[1,y,x]-((conv27[2,y,x])**2)

      o2[0, y,x] = conv15[0,y,x]* conv9[1,y,x]-((conv9[2,y,x])**2)
      o2[1, y,x] = conv27[0,y,x]*conv15[1,y,x]-((conv15[2,y,x])**2)
      o2[2, y,x] = conv39[0,y,x]*conv21[1,y,x]-((conv21[2,y,x])**2)
      o2[3, y,x] = conv51[0,y,x]*conv27[1,y,x]-((conv27[2,y,x])**2)

      o3[0, y,x] = conv27[0,y,x]* conv9[1,y,x]-((conv9[2,y,x])**2)
      o3[1, y,x] = conv51[0,y,x]*conv15[1,y,x]-((conv15[2,y,x])**2)
      o3[2, y,x] = conv75[0,y,x]*conv21[1,y,x]-((conv21[2,y,x])**2)
      o3[3, y,x] = conv99[0,y,x]*conv27[1,y,x]-((conv27[2,y,x])**2)

  extrema_points_1_1 = []
  extrema_points_1_2 = []
  extrema_points_2_1 = []
  extrema_points_2_2 = []
  extrema_points_3_1 = []
  extrema_points_3_2 = []
  #extrema_points_4 = [] 
  #print(clear,"\r", end="")

  #print("Process: Finding points for first octave","\r", end="")

  # Perform non maximal supression on determinant of Hessian.
  passedmax, passeddxhat, rejectedxhat = 0,0,0
  passedmax1, passeddxhat1, rejectedxhat1 = 0,0,0

  for y in range(0,I_bw.shape[0]):
    for x in range(0,I_bw.shape[1]):
      Flag = False
      if find_max_new(o1,1,y,x) == 1:
        if (accurate_keypoint(I_bw[y,x],conv9[:,y,x],conv15[:,y,x],conv21[:,y,x], 0.5) == 1):
          extrema_points_1_1.append([y,x,(2.0)])
          #I[y,x] = (0,0,255)
          cv2.circle(I,(x,y), 3, (0,0,255), -1)
        rejectedxhat += 1
        Flag = True
      if Flag == False and find_max_new(o1,2,y,x) == 1:
        if (accurate_keypoint(I_bw[y,x],conv15[:,y,x],conv21[:,y,x],conv27[:,y,x], 0.5) == 1):
          extrema_points_1_2.append([y,x,(2.8)])
          #I[y,x] = (0,0,255)
          cv2.circle(I,(x,y), 5, (0,0,155), -1)

  dogn1 = numpy.array(extrema_points_1_1)
  dogn2 = numpy.array(extrema_points_1_2)
  if (len(dogn1) > 0) and (len(dogn2)>0):
    result = numpy.vstack([dogn1, dogn2])
  #print("dog1")
  #print(dogn1)
  #print("\ndogn2")
  #print(dogn2)
  print ("Number of points in first octave: %d" % len(result))
  
  for y in range(0,I_bw.shape[0]):
    for x in range(0,I_bw.shape[1]):
      Flag = False
      if find_max_new(o2,1,y,x) == 1:
        if (accurate_keypoint(I_bw[y,x],conv15[:,y,x],conv27[:,y,x],conv39[:,y,x], 0.5) == 1):
          extrema_points_2_1.append([y,x,(3.6)])
          #I[y,x] = (0,0,255)
          cv2.circle(I,(x,y), 5, (0,255,0), 2)

        Flag = True
      if Flag == False and find_max_new(o2,2,y,x) == 1:
        if (accurate_keypoint(I_bw[y,x],conv27[:,y,x],conv39[:,y,x],conv51[:,y,x], 0.5) == 1):
          extrema_points_2_2.append([y,x,(5.2)])
          #I[y,x] = (0,0,255)
          cv2.circle(I,(x,y), 5, (0,155,0), 2)
  dogn3 =  numpy.array(extrema_points_2_1)
  dogn4 = numpy.array(extrema_points_2_2)
  if (len(dogn3) > 1) and (len(dogn4)>1):
    result1 = numpy.vstack([dogn3, dogn4])
    print ("Number of points in second octave: %d" % len(result1))

  for y in range(0,I_bw.shape[0]):
    for x in range(0,I_bw.shape[1]):
      Flag = False
      if find_max_new(o3,1,y,x) == 1:
        if (accurate_keypoint(I_bw[y,x],conv27[:,y,x],conv51[:,y,x],conv75[:,y,x], 0.5) == 1):
          #I[y,x] = (255,0,0)
          cv2.circle(I,(x,y), 7, (255,0,0), 2)
          extrema_points_3_1.append([y,x,(6.8)])
        Flag = True
      if Flag == False and find_max_new(o3,2,y,x) == 1:
        if (accurate_keypoint(I_bw[y,x],conv51[:,y,x],conv75[:,y,x],conv99[:,y,x], 0.5) == 1):
          #I[y,x] = (0,0,255)
          cv2.circle(I,(x,y), 7, (155,0,0), 2)
          extrema_points_3_2.append([y,x,(10.0)])

  dogn5 =  numpy.array(extrema_points_3_1)
  dogn6 = numpy.array(extrema_points_3_2)
  if (len(dogn5) == 0):
    dogn5 =[0,0,0]
  if (len(dogn6) == 0):
    dogn6 = [0,0,0]
  result2 = numpy.vstack([dogn5, dogn6])
  print ("Number of points in thrid octave: %d" % len(result2))
  alloctaves = numpy.vstack([result,result1,result2]) 
  #III = cv2.imread(filename)
  #coordin = (numpy.array([alloctaves[:,1], alloctaves[:,0]]).astype(int)).T
  #III[coordin] = (0,0,255)
  #cv2.imwrite(str(filename)[:-4] + "-surfdet" + ".jpg", III)
  cv2.imwrite("xx" + str(filename)[:-4] + ".jpg", I)
  h.points_to_txt_3_points(alloctaves, "surfallpoints.txt", "\n")
  print("pic written")
  return alloctaves 

"""
p = findSurfPoints("markstor2-seg.jpg")
I = cv2.imread("markstor2-seg.jpg")
p = findSurfPoints("erimitage2.jpg")
I = cv2.imread("erimitage2.jpg")

I[p[:,0], p[:,1]] = (0,0,255)
cv2.imshow("image", I)
cv2.waitKey(0)
"""
