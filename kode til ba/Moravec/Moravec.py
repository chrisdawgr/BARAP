import cv2
import time
import numpy as np
import scipy
from scipy import ndimage
from scipy import misc

def moravec(Iname):
  """
  Input : name of image
  Output: interest points
  """
  I = cv2.imread(Iname)
  I_bw = ndimage.filters.gaussian_filter(cv2.imread(Iname, 0).astype(float), 4.52)
  I_bw = misc.imresize(I_bw, 25, 'bilinear').astype(float)
  I = np.array(I_bw)

  cornerness = []
  corners = []

  for y in range(3, len(I_bw) - 3):
    for x in range(3, len(I_bw[0]) - 3):
      m1 = moravec_window(I_bw, y, x, 1, -1)
      m2 = moravec_window(I_bw, y, x, 0, -1)
      m3 = moravec_window(I_bw, y, x, -1, -1)
      m4 = moravec_window(I_bw, y, x, -1, 0)
      m5 = moravec_window(I_bw, y, x, -1, 1)
      m6 = moravec_window(I_bw, y, x, 0, 1)
      m7 = moravec_window(I_bw, y, x, 1, 1)
      m8 = moravec_window(I_bw, y, x, 1, 0)
      cornerness.append([y, x, min(m1, m2, m3, m4, m5, m6, m7, m8)])

  for cornerlen in range(0, len(cornerness)):
    #print(cornerness[cornerlen][2])
    if (cornerness[cornerlen][2] > 1900):
      y = cornerness[cornerlen][0]
      x = cornerness[cornerlen][1]
      cv2.circle(I, (x, y), 3, 4, 2)
      corners.append([y, x, 11])
  corners = np.array(corners)
  np.save(Iname[:-4], corners)
  print("# points found ", len(corners))
  #cv2.imwrite("IMG_9382-points.jpg", I)
  

def moravec_window(a, y, x, u, v):
  a1 = (int(a[y + u + 1, x + v - 1]) - int(a[y + 1, x - 1]))**2
  a2 = (int(a[y + u + 0, x + v - 1]) - int(a[y + 0, x - 1]))**2
  a3 = (int(a[y + u - 1, x + v - 1]) - int(a[y - 1, x - 1]))**2
  a4 = (int(a[y + u - 1, x + v + 0]) - int(a[y - 1, x + 0]))**2
  a5 = (int(a[y + u - 1, x + v + 1]) - int(a[y - 1, x + 1]))**2
  a6 = (int(a[y + u + 0, x + v + 1]) - int(a[y + 0, x + 1]))**2
  a7 = (int(a[y + u + 1, x + v + 1]) - int(a[y + 1, x + 1]))**2
  a8 = (int(a[y + u + 1, x + v + 0]) - int(a[y + 1, x + 0]))**2
  amid = (int(a[y + u + 0, x + v + 0]) - int(a[y + 0, x - 0]))**2
  res = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8+ amid
  return(res)

moravec('IMG_9428.jpg')
moravec('IMG_9456.jpg')
moravec('IMG_9488.jpg')
moravec('IMG_9492.jpg')
#moravec('IMG_9381.jpg')
"""
moravec('IMG_9370.jpg')
moravec('IMG_9371.jpg')
moravec('IMG_9372.jpg')
moravec('IMG_9373.jpg')
moravec('IMG_9374.jpg')
moravec('IMG_9375.jpg')
moravec('IMG_9376.jpg')
moravec('IMG_9377.jpg')
moravec('IMG_9378.jpg')
moravec('IMG_9379.jpg')
moravec('IMG_9380.jpg')
moravec('IMG_9427.jpg')
moravec('IMG_9455.jpg')
moravec('IMG_9487.jpg')
moravec('IMG_9491.jpg')
"""
