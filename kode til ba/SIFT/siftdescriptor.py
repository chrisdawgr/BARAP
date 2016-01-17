import helperfunctions as h
import siftorientation as siftori
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

def is_inbound(point, region_nr, a, b, c):
  """
  input:  index of region, direction values
  output: d value
  """
  a1 = a[:2]
  b1 = b[:2]
  c1 = c[:2]
  w1 = np.linalg.norm(a1 - point) / 6
  w2 = np.linalg.norm(b1 - point) / 6
  w3 = np.linalg.norm(c1 - point) / 6
  bool1 = 1
  bool2 = 1
  bool3 = 1
  if (region_nr + a[2] < 0 or region_nr + a[2] > 15):
    w1 = 0
    bool1 = 0
  if (region_nr + b[2] < 0 or region_nr + b[2] > 15):
    w2 = 0
    bool2 = 0
  if (region_nr + c[2] < 0 or region_nr + c[2] > 15):
    w3 = 0
    bool3 = 0
  return(w1, w2, w3, bool1, bool2, bool3)


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
  return(theta)
 

def sift_descriptor(I_name, points_orientations):
  """
  Input : Image name I, points (y,x,sigma)
  output: points p, and features f 
  """
  k_con = math.sqrt(2)
  k = k_con
  """
  sigma1 = np.array([k_con/2, 1.0, k_con , k_con ** 2, k_con ** 3, \
                     k_con, k_con ** 2, k_con ** 3, k_con ** 4, k_con ** 5, \
                     k_con ** 3, k_con ** 4, k_con ** 5, k_con ** 6, k_con ** 7, \
                     k_con ** 5, k_con ** 6, k_con ** 7, k_con ** 8, k_con ** 9, \
                     k_con ** 10, k_con ** 11, k_con ** 12, k_con ** 13, k_con ** 14])
  """

  sigma1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * k ** 2, 1.6 * k ** 3, \
  1.6 * k, 1.6 * k ** 2, 1.6 * k ** 3, 1.6 * k ** 4, 1.6 * k ** 5, \
  1.6 * k ** 3, 1.6 * k ** 4, 1.6 * k ** 5, 1.6 * k ** 6, 1.6 * k ** 7, \
  1.6 * k ** 5, 1.6 * k ** 6, 1.6 * k ** 7, 1.6 * k ** 8, 1.6 * k ** 9, \
  1.6 * k ** 7, 1.6 * k ** 8, 1.6 * k ** 9, 1.6 * k ** 10, 1.6 * k ** 11])




  I = cv2.imread(I_name)
  I0 = cv2.imread(I_name, 0)
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
       o4sc[0], o4sc[1], o4sc[2], o4sc[3], o4sc[4]]
       

  window_size = 16
  gauss_window = h.gauss(window_size, 8)

  final_points = []
  orien_of_bin = []

  len_of_desc = len(points_orientations)

  mag_bin = np.zeros([len_of_desc, 16, 4, 4])
  ori_bin = np.zeros([len_of_desc, 16, 4, 4])
  final_desc = []

  incrementeur = 0
  for p in points_orientations:

    hold_mag = np.empty([window_size,window_size])
    hold_ori = np.empty([window_size,window_size])

    ip_window = h.create_window_angles(I[int(p[2])], p, window_size + 2) 

    # creates bin for each point
    for y in range(0, window_size):
      for x in range(0, window_size):
        magnitude_p = magnitude(ip_window, [y + 1, x + 1])
        orientation_p = orientation(ip_window, [y + 1, x + 1])
        hold_mag[y, x] = magnitude_p
        hold_ori[y, x] = orientation_p


    hold_mag = np.multiply(hold_mag, gauss_window)
    hold_mag = np.reshape(hold_mag, -1)
    hold_ori = np.reshape(hold_ori, -1)

    v = 0
    for w in range(0, 4):
      for i in range(0, 13, 4):
        for j in range(0, 4):
          for k in range(0, 4):
            mag_bin[incrementeur][v][j][k] = hold_mag[k + i + j * 16 + w * 64]
            ori = hold_ori[k + i + j * 16 + w * 64]
            ori_bin[incrementeur][v][j][k] = ori
        v += 1
    incrementeur += 1


  """
                          val
        t-3   t-2   t-1   t0  t1  t2  t3

        directions: first dir is pointing north, northeast, east...
  """
  directions = np.array([[-2.5,1.5, -4],[-2.5,5.5, -3],[1.5,5.5, 1],[5.5,5.5, 5], \
                        [5.5,2.5, 4],[-2.5,5.5, 3],[-2.5,1.5, -1],[-2.5,-2.5, -5]])

  mid = np.array([1.5, 1.5])

  incc = 0
  for i in range(0, len(mag_bin)):
    descriptor = np.zeros([16,8])
    holder_mag = mag_bin[i]
    holder_ori = ori_bin[i]
    for region in range(0, 16):
      for y in range(0, 4):
        for x in range(0, 4):
          coor = np.array([y, x])
          bin_nr = (holder_ori[region, y, x] / 45.0)
          if (bin_nr > 7):
            bin_nr = bin_nr - 1
          t1_nr = math.ceil(bin_nr) % 8

          tm1_nr = math.floor(bin_nr) % 8
          if (bin_nr % 1.0 == 0):
            tm1_nr = (bin_nr + 1) % 8 
            
          t2_nr = (t1_nr + 1) % 8
          tm2_nr = (tm1_nr - 1) % 8

          t0_weight = 1 - np.linalg.norm(mid - coor) / 2.7
          t0 = t0_weight * holder_mag[region, y, x]

          A = 0
          B = t0
          C = 0
          a = -t0/9
          b = 0
          c = t0
          offset = bin_nr % 1.0
          t1 = a * (1+offset)**2 + b * (1+offset) + c
          tm1 = a * (-1 + offset)**2 + b * (-1+offset) + c
          t2 = a * (2+offset)**2 + b * (2+offset) + c
          tm2 = a * (-2+offset)**2 + b * (-2+offset) + c
          descriptor[region, t1_nr] += t1
          descriptor[region, tm1_nr] += tm1
          descriptor[region, t2_nr] += t2
          descriptor[region, tm2_nr] += tm2

          # first quadrant
          if (y >= 0 and y <= 1 and x >= 2 and x <= 3):
            dir1 = directions[0]
            dir2 = directions[1]
            dir3 = directions[2]
            (w1,w2,w3,bool1,bool2,bool3) = is_inbound([y,x],region, dir1, dir2, dir3)

          if (y >= 0 and y <= 1 and x >= 0 and x <= 1):
            dir1 = directions[6]
            dir2 = directions[7]
            dir3 = directions[0]
            (w1,w2,w3,bool1,bool2,bool3) = is_inbound([y,x],region, dir1, dir2, dir3)

          if (y >= 2 and y <= 3 and x >= 0 and x <= 1):
            dir1 = directions[4]
            dir2 = directions[5]
            dir3 = directions[6]
            (w1,w2,w3,bool1,bool2,bool3) = is_inbound([y,x],region, dir1, dir2, dir3)

          if (y >= 2 and y <= 3 and x >= 2 and x <= 3):
            dir1 = directions[2]
            dir2 = directions[3]
            dir3 = directions[4]
            (w1,w2,w3,bool1,bool2,bool3) = is_inbound([y,x],region, dir1, dir2, dir3)


          descriptor[(region + dir1[2]) * bool1, t1_nr] += w1 * t1
          descriptor[(region + dir1[2]) * bool1, tm1_nr] += w1 * tm1
          descriptor[(region + dir1[2]) * bool1, t2_nr] += w1 * t2
          descriptor[(region + dir1[2]) * bool1, tm2_nr] += w1 * tm1

          descriptor[(region + dir2[2]) * bool2, t1_nr] += w2 * t1
          descriptor[(region + dir2[2]) * bool2, tm1_nr] += w2 * tm1
          descriptor[(region + dir2[2]) * bool2, t2_nr] += w2 * t2
          descriptor[(region + dir2[2]) * bool2, tm2_nr] += w2 * tm2

          descriptor[(region + dir3[2]) * bool3, t1_nr] += w3 * t1
          descriptor[(region + dir3[2]) * bool3, tm1_nr] += w3 * tm1
          descriptor[(region + dir3[2]) * bool3, t2_nr] += w3 * t2
          descriptor[(region + dir3[2]) * bool3, tm2_nr] += w3 * tm2
    final_desc.append(np.array(descriptor.reshape(-1)))
    incc += 1
  
  print("length of descriptor processed: ", incc)
  ret = []
  for vec_i in range(0, len(final_desc)):
    dd = final_desc[vec_i] / np.linalg.norm(final_desc[vec_i])
    dd = np.clip(dd, 0, 0.2)
    dd = dd / np.linalg.norm(dd)
    ret.append([points_orientations[vec_i], np.array(dd)])


  return(ret)
