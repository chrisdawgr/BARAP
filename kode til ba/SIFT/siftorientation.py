import helperfunctions as h
import scipy
from scipy import ndimage
from scipy import misc
import cv2
import math
import numpy as np


def magnitude(I, point):
  """
  Input : Image I, point point
  Output: outputs the magnitude
  """
  w = h.create_window(I, point, 3)
  mag = math.sqrt((w[1][0] - w[1][2])**2 + (w[0][1] - w[2][1])**2)
  return(mag)


def orientation(I, point):
  """
  Input : Image I, point point
  Output: orientation of image
  """
  w = h.create_window(I, point, 3)
  Ly = w[0][1] - w[2][1]
  Lx = w[1][0] - w[1][2]
  theta = 0.5 * math.pi - math.atan2(Lx, Ly)
  theta = math.degrees(theta % (2 * math.pi))
  #print(theta)
  return(theta)
 

def sift_orientation(I_name, points):
  window_size = 16
  """
  Input : image I, interest points, size of window
  Output: assigns an orientation to the interest points
  """
  k_con = math.sqrt(2)
  k = math.sqrt(2)
  """
  sigma1 = np.array([k_con/2, 1.0, k_con , k_con ** 2, k_con ** 3, \
                     k_con, k_con ** 2, k_con ** 3, k_con ** 4, k_con ** 5, \
                     k_con ** 3, k_con ** 4, k_con ** 5, k_con ** 6, k_con ** 7, \
                     k_con ** 5, k_con ** 6, k_con ** 7, k_con ** 8, k_con ** 9])
  """

  sigma1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * k ** 2, 1.6 * k ** 3, \
  1.6 * k, 1.6 * k ** 2, 1.6 * k ** 3, 1.6 * k ** 4, 1.6 * k ** 5, \
  1.6 * k ** 3, 1.6 * k ** 4, 1.6 * k ** 5, 1.6 * k ** 6, 1.6 * k ** 7, \
  1.6 * k ** 5, 1.6 * k ** 6, 1.6 * k ** 7, 1.6 * k ** 8, 1.6 * k ** 9, \
  1.6 * k ** 7, 1.6 * k ** 8, 1.6 * k ** 9, 1.6 * k ** 10, 1.6 * k ** 11])

  I = cv2.imread(I_name).astype(float)
  I_show = cv2.imread(I_name).astype(float)

  I0 = cv2.imread(I_name, 0).astype(float)
  I1 = misc.imresize(I0, 50, 'bilinear').astype(float)
  I2 = misc.imresize(I0, 25, 'bilinear').astype(float)
  I3 = misc.imresize(I0, 1/8.0, 'bilinear').astype(float)
  I4 = misc.imresize(I0, 1/16.0, 'bilinear').astype(float)

  o0sc = np.zeros((5, I0.shape[0],I0.shape[1]))
  o1sc = np.zeros((5, I1.shape[0],I1.shape[1]))
  o2sc = np.zeros((5, I2.shape[0],I2.shape[1]))
  o3sc = np.zeros((5, I3.shape[0],I3.shape[1]))
  o4sc = np.zeros((5, I4.shape[0],I4.shape[1]))

  for i in range(0,5):
    """
    o1sc[i,:,:] = cv2.filter2D(I0, -1, h.gauss(9, sigma1[i]))
    o2sc[i,:,:] = misc.imresize(cv2.filter2D(I0, -1, h.gauss(9,sigma1[i + 5])), 50, 'bilinear').astype(float)
    o3sc[i,:,:] = misc.imresize(cv2.filter2D(I0, -1, h.gauss(9,sigma1[i + 10])), 25, 'bilinear').astype(float)
    o4sc[i,:,:] = misc.imresize(misc.imresize(cv2.filter2D(I0, -1, h.gauss(9,sigma1[i + 15])), 25, 'bilinear').astype(float), 50, 'bilinear').astype(float)
    """

    o0sc[i,:,:] = ndimage.filters.gaussian_filter(I0, sigma1[i])
    o1sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma1[i]), 50, 'bilinear')
    o2sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma1[i]), 25, 'bilinear')
    o3sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma1[i]), 1/8.0, 'bilinear')
    o4sc[i,:,:] = misc.imresize(ndimage.filters.gaussian_filter(I0, sigma1[i]), 1/16.0, 'bilinear')

  I = [o0sc[0], o0sc[1], o0sc[2], o0sc[3], o0sc[4], o1sc[0], o1sc[1], o1sc[2], o1sc[3], o1sc[4], \
       o2sc[0], o2sc[1], o2sc[2], o2sc[3], o2sc[4], o3sc[0], o3sc[1], o3sc[2], o3sc[3], o3sc[4], \
       o4sc[1], o4sc[2], o4sc[3], o4sc[4], o4sc[4]]
       

  # point with orientation; [y, x, s, o]

  final_points = []
  ret_points = []
  

  # length of outer list
  o = 0
  for p in points:
    counter = 1
    max_size_y = len(I[int(p[2])]) - (window_size)
    max_size_x = len(I[int(p[2])][0]) - (window_size)
    min_size_y = window_size
    min_size_x = window_size
    if ((p[0] < max_size_y) and (min_size_y < p[0]) and \
        (p[1] < max_size_x and min_size_x < p[1])):
      if (p[2] > 5 and p[2] < 8):
        cv2.circle(I_show, (int(p[1] * 2), int(p[0] * 2)), 12, (0,255,0),2)
      if (p[2] > 10 and p[2] < 13):
        cv2.circle(I_show, (int(p[1] * 4), int(p[0] * 4)), 24, (255,0,0),2)
      if (p[2] > 15 and p[2] < 18):
        cv2.circle(I_show, (int(p[1] * 8), int(p[0] * 8)), 30, (125,125,0),2)

      final_points.append(p)
      o += 1
  cv2.imwrite("points.jpg", I_show)


  # size of all usable points o

  print("length of usable points, orientation", o)
  i = 0
  for p in final_points:
    bins = np.zeros([36])
    window_size = 16
    gauss_window = h.gauss(window_size, 1.5 * p[2])
    ip_window = h.create_window(I[int(p[2])], p, window_size + 2) 
    orien_of_bin = []

    # creates bin for each point
    hold_mag = np.empty((window_size, window_size))
    hold_ori = np.empty((window_size, window_size))
    for y in range(0, window_size):
      for x in range(0, window_size):
        magnitude_p = magnitude(ip_window, [y + 1, x + 1])
        orientation_p = orientation(ip_window, [y + 1, x + 1])
        orientation_p = math.floor(orientation_p/10.0)
        hold_mag[y, x] = magnitude_p
        hold_ori[y, x] = orientation_p

    hold_mag = np.multiply(hold_mag, gauss_window)
    hold_mag = np.reshape(hold_mag, -1)
    hold_ori = np.reshape(hold_ori, -1)

    for j in range(0, len(hold_ori)):
      bins[hold_ori[j]] += hold_mag[j]

      
    # index of max element in bin
    max_index = bins.argmax()
      
    max_val = bins[max_index]


    for j in range(0, 35):
      if (bins[j] >= max_val * 0.8):
        orien_of_bin.append(j)

    new_orien = list(orien_of_bin)

    cc = 0
    for i in new_orien:
      if (i == 1):
        A = 0
        B = bins[i] 
        C = bins[i + 1]
      if (i == 35):
        A = bins[i-1]
        B = bins[i]
        C = 0
      else:
        A = bins[i - 1]
        B = bins[i]
        C = bins[i + 1]
      a = A + (C-A)/2.0 - B
      b = (C-A)/2.0
      c = B
      if (b == 0 or a == 0):
        toppoint = 0
      else: 
        toppoint = -b / (2 * a)
      degree = (toppoint * 10) + i * 10
      #print(degree)

      if (degree < 0):
        degree = 360.0 + degree
      ret_points.append([p[0], p[1], p[2], degree])
      cc += 1

      if (cc == 1):
        break

    counter += 1
  
  ret_points = np.array(ret_points)
  return (ret_points)

#harrisPoints = np.load("IMG_9374.npy")
#sift_orientation("IMG_9374.jpg", harrisPoints)
