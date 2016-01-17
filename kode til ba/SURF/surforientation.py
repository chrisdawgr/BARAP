import numpy as np
import cv2
import helperfunctions as hh
import math
import surfdescriptor as h2



def surf_orientation(I_name, point):
  """
  input: (y, x, s), integral_img for scale s
  output : (y, x, s, o)
  """
  counter = 1
  final_keypoints = []
  orientations = []
  I_int = cv2.imread(I_name, 0).astype(float)

  scale_pics = [cv2.filter2D(I_int, -1, hh.gauss(9, 1.2)), cv2.filter2D(I_int, -1, hh.gauss(15, 1.2)), \
                cv2.filter2D(I_int, -1, hh.gauss(21, 1.2)), cv2.filter2D(I_int, -1, hh.gauss(27, 1.2)), \
                cv2.filter2D(I_int, -1, hh.gauss(39, 1.2)), cv2.filter2D(I_int, -1, hh.gauss(75, 1.2))]
  scale_pics = [cv2.integral(scale_pics[0]), cv2.integral(scale_pics[1]), \
                cv2.integral(scale_pics[2]), cv2.integral(scale_pics[3]), \
                cv2.integral(scale_pics[4]), cv2.integral(scale_pics[5])]

  for p_a in point:
    hs = h2.haar_size(p_a[2])
    max_size_y = len(I_int) - (20 * p_a[2] + hs) 
    max_size_x = len(I_int[0]) - (20 * p_a[2] + hs) 
    min_size_x = min_size_y = h2.round_four(20 * p_a[2] ) + hs 

    if ((p_a[0] < max_size_y) and (min_size_y < p_a[0]) and \
        (p_a[1] < max_size_x) and (min_size_x < p_a[1])):
      #cv2.circle(I, (p[1].astype(int), p[0].astype(int)), int(round(p[2])), (0,0,255), 3)
      final_keypoints.append([p_a[0], p_a[1], p_a[2]])

  #keypoint area = [y, x, s]
  for p_a in final_keypoints:
    window_size = (int(6 * p_a[2]) / p_a[2]) + 1
    d_x = np.zeros([window_size, window_size])
    d_y = np.zeros([window_size, window_size])

    if (p_a[2] > 1.9 and p_a[2] < 2.1):
      I_int = scale_pics[0]
    if (p_a[2] > 2.7 and p_a[2] < 2.9):
      I_int = scale_pics[1]
    if (p_a[2] > 3.5 and p_a[2] < 3.6):
      I_int = scale_pics[2]
    if (p_a[2] > 5.1 and p_a[2] < 5.3):
      I_int = scale_pics[3]
    if (p_a[2] > 6.7 and p_a[2] < 6.9):
      I_int = scale_pics[4]
    if (p_a[2] > 9.9 and p_a[2] < 10.1):
      I_int = scale_pics[5]


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
    # Calculating the haar wavelet response, in a 20x20 window
    """

    haar_hs = int(window_size / 2)
    incre = int(p_a[2])
    
    #y1 = - haar_hs * incre + int(p_a[0])
    y1 = 0
    for y in range(int(p_a[0]) - (haar_hs * incre), int(p_a[0]) + (haar_hs * incre), incre):
      #x1 = - haar_hs * incre + int(p_a[1])
      x1 = 0
      for x in range(int(p_a[1]) - (haar_hs * incre), int(p_a[1]) + (haar_hs * incre), incre):
        if (y1 < 0 or x1 < 0):
          print(y1,x1)
        #print("y = " + str(y), "x = " + str(x), "scale = " + str(p_a[2]), "hs = " + str(hs))
        a = I_int[y - haar_hs, x - haar_hs]
        b = I_int[y - haar_hs, x + haar_hs]
        c = I_int[y + haar_hs, x - haar_hs]
        d = I_int[y + haar_hs, x + haar_hs]
        e = I_int[y, x - haar_hs]
        f = I_int[y, x + haar_hs]
        g = I_int[y - haar_hs, x]
        h = I_int[y + haar_hs, x]
        d_x[y1, x1] = (h + a - c - g) - (d + g - h - b)
        d_y[y1, x1] = (f + a - e - b) - (d + e - c - f)
        x1 += 1
      y1 += 1
    #y1 = - haar_hs * incre + int(p_a[0])
    #for y in range(0, window_size, int(incre)):
      #x1 = - haar_hs * incre + int(p_a[1])
      #for x in range(0, window_size, int(incre)):
        #x1 += incre
      #y1 += incre
    d_x = np.multiply(hh.gauss(len(d_x), 2.5 * p_a[2]), d_x)
    d_y = np.multiply(hh.gauss(len(d_y), 2.5 * p_a[2]), d_y)

    d_x = hh.circle_matrix(d_x)
    d_y = hh.circle_matrix(d_y)
    np.set_printoptions(precision=0)
    #print("\n")
    #print(d_x)
    #print("\n")
    #print(d_y)
    #print("\n")
    
    
    dirs = np.zeros([6, 2])
    for y in range (0, len(d_x)):
      for x in range(0, len(d_x)):
        dirs[math.floor(hh.orientation([d_x[y,x], d_y[y,x]]) / 60.0), 0] += d_x[y,x]
        dirs[math.floor(hh.orientation([d_x[y,x], d_y[y,x]]) / 60.0), 1] += d_y[y,x]

    #print(dirs)
    #print("\n")
    #print("\n")
    #print("\n")
    #print("\n")

    orientations.append(hh.orientation(dirs[np.argmax(np.sum(np.abs(dirs), axis=1))]))

  ret_points = []
  for i in range(0, len(final_keypoints)):
    ret_points.append([final_keypoints[i][0], final_keypoints[i][1], final_keypoints[i][2], orientations[i]])

  return(ret_points)
