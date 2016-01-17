import math
import cv2
import numpy as np
import helperfunctions as h2

# if (i == 0) = dx, else dy
def create_haar_window(window_size, j):
  """
  input: size of the window, i=0 = dx, i=1 = dy
  output: a haarwindow, of size:  i + (i % 2)
  """
  i = round(j)
  window_size = int(round(window_size))
  round_size = (window_size + window_size % 2)
  haar_window = np.zeros([round_size, round_size])
  half_len = round_size / 2
  if (i == 0):
    for y in range(0, round_size):
      for x in range(0, round_size):
        if (x + 1 > half_len):
          haar_window[y, x] = 1
        else:
          haar_window[y, x] = -1
  
  else:
   for y in range(0, round_size):
        for x in range(0, round_size):
          if (y + 1 > half_len):
            haar_window[y, x] = 1
          else:
            haar_window[y, x] = 0

  return(haar_window)


def haar_size(s):
  return(int(round(s * 2) + (round(2 * s) % 2)))

def round_four(a):
  """
  input: size of the number to round
  output: rounds to nearest 5
  """

  mod_res = a % 4
  if(mod_res > 2):
    return(int(a + (4 - mod_res)))
  else:
    return(int(a - mod_res))


def surf_descriptor_w_orientation(I_name, keypoints):

  """
  input: [y, x, s, o], name of picture
  output: Descriptor
  """
  I_bw = cv2.imread(I_name, 0)
  I = cv2.imread(I_name)
  I_int = cv2.integral(I_bw)
  keypoint_descriptor = []
  final_keypoints = []
  print("starting descriptor")

  # Removing keypoints if the keypoint collecting region is outside the image
  for p_a in keypoints:
    hs = haar_size(p_a[2])
    max_size_y = len(I_bw) - (20 * p_a[2] + hs)
    max_size_x = len(I_bw[0]) - (20 * p_a[2] + hs)
    min_size_x = min_size_y = round_four(20 * p_a[2]) + hs
    #print("maxsize y = " + str(max_size_y), "maxsize x " + str(max_size_x), "minsizex/y = " + str(min_size_y))
    if ((p_a[0] < max_size_y) and (min_size_y < p_a[0]) and \
        (p_a[1] < max_size_x) and (min_size_x < p_a[1])):
      cv2.circle(I, (p_a[1].astype(int), p_a[0].astype(int)), int(round(p_a[2])), (0,0,255), 3)
      final_keypoints.append([p_a[0], p_a[1], p_a[2], p_a[3]])

  cv2.imwrite("zzdes-" + I_name[:-4] + ".jpg", I)
  print("length of final_keypoints:" + str(len(final_keypoints)))
  
  #keypoint area = [y, x, s]
  counter = 1
  for p_a in final_keypoints:
    col, row = np.shape(I_bw)
    rotmat = cv2.getRotationMatrix2D((p_a[1], p_a[0]), p_a[3], 1)
    dst = cv2.warpAffine(I_bw, rotmat, (int(row), int(col)))
    I_int = cv2.integral(dst)
    print(counter)
    counter += 1

    region_size = round_four(20 * p_a[2])
    sub_region_size = region_size/4.0
    descriptor = np.zeros([64, 1])
    d_x = np.zeros([20,20])
    d_y = np.zeros([20,20])


    """   
      a   g   b
        ######
        ######
      e ##p### f
        ###### 
        ######
        ######
       c  gh  d 
    the integral square aecg subtracted from fbhd gives the dx-haar-wavelet response 
    """   
    # Calculating the haar wavelet response, in a 20x20 window

    haar_hs = haar_size(p_a[2]) / 2
    incre = (sub_region_size/5.0)/5.0

    y1 = -region_size/2 + int(p_a[0])
    for y in range(0, 20):
      x1 =-region_size/2 + int(p_a[1])
      for x in range(0, 20):
        #print(y1, region_size/2 + int(p_a[1]))
        #print(y1- haar_hs + 1, x1 + haar_hs - 1)
        #print("y = " + str(y1), "x = " + str(x1), "scale = " + str(p[2]), "hs = " + str(hs))
        #print("y = " + str(I_int[y1,x1]))
        #print(haar_hs, incre)
        if (y1 < 0 or x1 < 0):
          print(y1,x1)
          print("y="+str(p_a[0]),"x="+str(p_a[1]), "scale="+str(p[2]))
        a = I_int[y1 - haar_hs, x1 - haar_hs]
        b = I_int[y1 - haar_hs, x1 + haar_hs]
        c = I_int[y1 + haar_hs, x1 - haar_hs]
        d = I_int[y1 + haar_hs, x1 + haar_hs]
        e = I_int[y1, x1 - haar_hs]
        f = I_int[y1, x1 + haar_hs]
        g = I_int[y1 - haar_hs, x1]
        h = I_int[y1 + haar_hs, x1]
        d_x[y, x] = (h + a - c - g) - (d + g - h - b)
        d_y[y, x] = (f + a - e - b) - (d + e - c - f)
        x1 += incre
      y1 += incre

    d_x = np.multiply(h2.gauss(len(d_x), 3.3 * p_a[2]), d_x)
    d_y = np.multiply(h2.gauss(len(d_y), 3.3 * p_a[2]), d_y)
    #if ((p_a[0] == 712 and p_a[1] == 1514) or (p_a[0] == 530 and p_a[1] == 327)):
    #  np.set_printoptions(precision=5)
    #  print(d_x)
    #  print(d_y)

    i = 0
    for y in range(0, 4):
      for x in range(0, 4):
        field_dx = np.sum(d_x[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5])
        field_dy = np.sum(d_y[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5])
        field_abs_dx = np.sum(np.abs(d_x[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5]))
        field_abs_dy = np.sum(np.abs(d_y[y * 5 : (y+1) * 5, x * 5 : (x+1) * 5]))

        descriptor[i + 0] = field_dx
        descriptor[i + 1] = field_dy
        descriptor[i + 2] = field_abs_dx
        descriptor[i + 3] = field_abs_dy
        i += 4

    keypoint_descriptor.append([p_a[:-1], descriptor / np.linalg.norm(descriptor)])

  #print("length of keypoint descriptor: " + str(len(keypoint_descriptor)))
  #print("\n")


  return(keypoint_descriptor)
