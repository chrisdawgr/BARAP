import cv2
import matplotlib.pyplot as plt
import math
import random
import numpy as np

def gauss(size, sigma):
  """
  Creates Gusssian kernel
  """
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / (size-1.0) 
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      frac = (1.0/(2.0 * math.pi * sigma**2)) 
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(abs(gauss_kernel)))
  return(gauss_kernel)

def gaussdx(size, sigma):
  """
  Creates a kernel with the first derivative of gauss
  """
  #NOTE: Guys at NVidia suggests a gausskernel of size 3*sigma
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / (size-1.0) 
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      gauss_calc = (((-x1/(2.0*math.pi*sigma**4.0)*(math.exp(-((x1**2.0+y1**2.0)/(2.0*sigma**2.0)))))))
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(abs(gauss_kernel)))
  return(gauss_kernel)

def gaussdy(size, sigma):
  return np.transpose((gaussdx(size,sigma)))

def gauss2x(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / (size-1.0) 
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      frac = -1*((x1**2-sigma**2)/(2.0 * math.pi * sigma**4.0))
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(abs(gauss_kernel)))
  return(gauss_kernel)

def gauss2y(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / (size-1.0) 
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      frac = -1*((y1**2-sigma**2)/(2.0 * math.pi * sigma**6.0))
      exponent = (x1**2.0 + y1**2.0)/(2.0 * sigma**2.0)
      gauss_calc = frac * math.exp(- exponent)
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(np.abs(gauss_kernel)))
  return(gauss_kernel)

def gauss2xy(size, sigma):
  D = np.zeros([size, size])
  gauss_kernel = np.zeros([size, size])
  stepsize = 9.0 / (size-1.0) 
  y1 = -4.5
  for y in range(0, size):
    x1 = -4.5
    for x in range(0, size):
      first = ((x1*y1)/2.0*math.pi**6.0)
      second = math.exp(-((x1**2.0+y1**2.0)/(2.0*sigma**2.0)))
      gauss_calc = first*second
      gauss_kernel[y][x] = gauss_calc
      x1 += stepsize
    y1 += stepsize
  gauss_kernel = gauss_kernel / sum(sum(np.abs(gauss_kernel)))
  return(gauss_kernel)


def create_window(I, point, window_size):
  """
  Input : Image, [y,x], size of window
  Output: Creates a window with size window_size of I, and makes point middle
  """
  D = np.empty([window_size, window_size])
  y_p = point[0]
  x_p = point[1]

  for y in range(-window_size/2, window_size/2):
    for x in range(-window_size/2, window_size/2):
      D[y][x] = I[y_p + y][x_p  + x]

  return(D)

def create_window_angles(I, point, window_size):
  """
  Input : Image, [y,x], size of window
  Output: Creates a window with size window_size of I, and makes point middle
  """
  D = np.empty([window_size, window_size])
  y_p = point[0]
  x_p = point[1]

  #delete_this = []

  for y in range(-window_size/2, window_size/2):
    for x in range(-window_size/2, window_size/2):
      theta = point[3] * math.pi / 180.0
      xrot = round((math.cos(theta) * y) - (math.sin(theta) * x))
      yrot = round((math.sin(theta) * y) + (math.cos(theta) * x))

      D[y][x] = I[y_p  + yrot ][x_p + xrot ]

      #print(      y_p  + yrot , x_p + xrot )
      #delete_this.append([y_p + yrot, x_p + xrot])
  #delete_this = np.array(delete_this)
  #plt.scatter(delete_this[:,0], delete_this[:,1])
  #plt.show()

  return(D)


def points_to_txt(points, filename_out, seperate_by):
  """
  input = points, filename to output, how to seperate the lists, eg "\n", "\t", "\n\n" etc
  output = file with filename_out with the points
  """
  file_o = open(filename_out, 'w')

  for i in points:
    file_o.write(str(i[0]) + " " + str(i[1]))
    file_o.write(seperate_by)
  file_o.close()

def points_to_txt_3_points(points, filename_out, seperate_by):
  """
  input = points, filename to output, how to seperate the lists, eg "\n", "\t", "\n\n" etc
  output = file with filename_out with the points
  """
  file_o = open(filename_out, 'w')

  for i in points:
    file_o.write(str(i[0]) + " " + str(i[1]) + " " + str(i[2]))
    file_o.write(seperate_by)
  file_o.close()


def points_to_txt2(points, filename_out, seperate_by):
  """
  input = points, filename to output, how to seperate the lists, eg "\n", "\t", "\n\n" etc
  output = file with filename_out with the points
  """
  file_o = open(filename_out, 'w')

  for i in points:
    #print(i)
    file_o.write(str(i))
    file_o.write(seperate_by)
  file_o.close()


def txt_to_points(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()
  for i in range(0, len(points_str), 2):
    result.append([int(points_str[i]), int(points_str[i + 1])])
  return result


def txt_to_3_points(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()
  for i in range(0, len(points_str), 3):
    result.append([int(points_str[i]), int(points_str[i + 1]), int(points_str[i + 2])])
  return result


def txt_to_3_points_float(filename):
  result = []
  oo = open(filename, "r")
  points_str = oo.read()
  oo.close()
  points_str = points_str.split()
  for i in range(0, len(points_str), 3):
    result.append([int(points_str[i]), int(points_str[i + 1]), float(points_str[i + 2])])
  return result


def matrix_substraction(m1, m2):
  dim = m1.shape
  height = dim[0]
  length = dim[1]
  mat = np.zeros([height, length], dtype='uint8')
  for y in range (0, height):
    for x in range(0, length):
      if (m1[y][x] < m2[y][x]):
        mat[y][x] = 0
      else:
        mat[y][x] = m1[y][x] - m2[y][x]
  return(mat)

def color_pic(*arg):
  """ 
  input: I, points
  output: pic
  """
  I = arg[0]
  points = arg[1]

  if (len(arg) >= 2):
    for p in points:
      I[p[0]][p[1]] = [0,0,255]
    cv2.imshow('image', I)
    cv2.waitKey(0)

  if (len(arg) == 3):
    name = arg[2]
    cv2.imwrite(name, I)

def color_scale(scale1):
    if (scale1 > 1.9 and scale1 < 2.1):
      return (0, 0, 255)
    if (scale1 > 2.7 and scale1 < 2.9):
      return (0, 0, 155)
    if (scale1 > 3.5 and scale1 < 3.6):
      return (0, 255, 0)
    if (scale1 > 5.1 and scale1 < 5.3):
      return (0, 155, 0)
    if (scale1 > 6.7 and scale1 < 6.9):
      return (255, 0, 0)
    if (scale1 > 9.9 and scale1 < 10.1):
      return (155, 0, 0)


def drawMatches(I1, kp1, I2, kp2, matches):
  """
  img1,img2 - Grayscale images
  kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
            detection algorithms
  matches - A list of matches of corresponding keypoints through any
            OpenCV keypoint matching algorithm
  """
  print("drawing matches")
  img1 = cv2.imread(I1, 0)
  img2 = cv2.imread(I2, 0)

  # Create a new output image that concatenates the two images together
  # (a.k.a) a montage
  rows1 = len(img1)
  cols1 = len(img1[0])
  rows2 = len(img2)
  cols2 = len(img2[0])

  out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

  # Place the first image to the left
  out[:rows1,:cols1] = np.dstack([img1, img1, img1])

  # Place the next image to the right of it
  out[:rows2,cols1:] = np.dstack([img2, img2, img2])

  # For each pair of points we have between both images
  # draw circles, then connect a line between them
  for mat in range(0, len(kp1)):

    # Get the matching keypoints for each of the images
    # x - columns
    # y - rows
    (y1,x1,scale1) = (kp1[mat])
    (y2,x2,scale2) = (kp2[mat])

    color1 = color_scale(scale1)
    color2 = color_scale(scale2)

    # Draw a small circle at both co-ordinates
    # radius 4
    # colour blue
    # thickness = 1
    cv2.circle(out, (int(x1),int(y1)), 5 * int(scale1), color1, 3)   
    cv2.circle(out, (int(x2)+cols1,int(y2)), 5 * int(scale2), color2, 3)

    # Draw a line in between the two points
    # thickness = 1
    # colour blue

    if (scale1 != scale2):
      cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (125,0,125), 3)
    else:
      cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color1, 3)


  # Show the image
  #cv2.imshow('Matched Features', out)
  cv2.imwrite(str(I1) + "-advanced-matching.jpg", out)
  cv2.waitKey(0)
  cv2.destroyWindow('Matched Features')

  # Also return the image if you'd like a copy
  return out

def oneNN(descs1, descs2, p1, p2):
  print("calculating oneNN")
  print(len(descs1), len(p1))
  print(len(descs2), len(p2))

  res_1 = []
  res_2 = []
  res_d_1 = []
  res_d_2 = []
  descs1 = np.array(descs1)
  descs2 = np.array(descs2)

  for i_desc1 in range(0, len(descs1)):
    desc1 = descs1[i_desc1]
    s_dist = float("inf")
    s_dist_index = 0
    for i_desc2 in range(0, len(descs2)):
      desc2 = descs2[i_desc2]
      dist = np.linalg.norm(desc2 - desc1)

      if (dist < s_dist):
        s_dist = dist
        s_dist_index = i_desc2

    res_1.append(p1[i_desc1])
    res_d_1.append(desc1)
    res_2.append(p2[s_dist_index])
    res_d_2.append(descs2[s_dist_index])

  """
  I = cv2.imread("room10.jpg")
  I2 = cv2.imread("room11.jpg")
  for fin in range(0, len(res_1)):
    I[res_1[fin][0], res_1[fin][1]] = (0,0,255)
    I2[res_2[fin][0], res_2[fin][1]] = (0,0,255)
    cv2.circle(I, (res_1[fin][1].astype(int), res_1[fin][0].astype(int)), 10, (0,255,0), 3)
    cv2.circle(I2, (res_2[fin][1].astype(int), res_2[fin][0].astype(int)), 10, (0,255,0), 3)

  cv2.imwrite("zzonenn" + "room10" + ".jpg", I)
  cv2.imwrite("zzonenn" + "room11" + ".jpg", I2)

  """
  #res_1 = np.array(res_1)
  #res_2 = np.array(res_2)
  #res_d_1 = np.array(res_d_1)
  #res_d_2 = np.array(res_d_2)
  return(res_1, res_2, res_d_1, res_d_2)



def drawMatches2(I1, kp1, I2, kp2, matches):
  """
  img1,img2 - Grayscale images
  kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
            detection algorithms
  matches - A list of matches of corresponding keypoints through any
            OpenCV keypoint matching algorithm
  """
  print("drawing matches")
  img1 = cv2.imread(I1, 0)
  img2 = cv2.imread(I2, 0)

  # Create a new output image that concatenates the two images together
  # (a.k.a) a montage
  rows1 = len(img1)
  cols1 = len(img1[0])
  rows2 = len(img2)
  cols2 = len(img2[0])

  out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

  # Place the first image to the left
  out[:rows1,:cols1] = np.dstack([img1, img1, img1])

  # Place the next image to the right of it
  out[:rows2,cols1:] = np.dstack([img2, img2, img2])

  # For each pair of points we have between both images
  # draw circles, then connect a line between them
  for mat in range(0, len(kp1)):

    # Get the matching keypoints for each of the images
    # x - columns
    # y - rows
    (y1, x1, scale1) = (kp1[mat])
    (y2, x2, scale2) = (kp2[mat])

    color1 = color_scale(scale1)
    color2 = color_scale(scale2)

    # Draw a small circle at both co-ordinates
    # radius 4
    # colour blue
    # thickness = 1
    cv2.circle(out, (int(x1),int(y1)), 5 * int(scale1), color1, 3)   
    cv2.circle(out, (int(x2)+cols1,int(y2)), 5 * int(scale2), color2, 3)

    # Draw a line in between the two points
    # thickness = 1
    # colour blue

    if (scale1 != scale2):
      cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (125,0,125), 3)
    else:
      cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), color1, 3)

  # Show the image
  #cv2.imshow('Matched Features', out)
  cv2.imwrite(str(I1) + "-advanced-matching.jpg", out)
  #cv2.waitKey(0)
  #cv2.destroyWindow('Matched Features')

  # Also return the image if you'd like a copy
  return out




def oneNN_wdist(descs1, descs2, p1, p2):
  print("calculating oneNN sorted after euclidean distance")
  print(len(descs1), len(p1))
  print(len(descs2), len(p2))

  res_1 = []
  res_2 = []
  res_d_1 = []
  res_d_2 = []
  res_d_d = []
  descs1 = np.array(descs1)
  descs2 = np.array(descs2)
  distances = []

  for i_desc1 in range(0, len(descs1)):
    desc1 = descs1[i_desc1]
    s_dist = float("inf")
    s_dist_index = 0
    for i_desc2 in range(0, len(descs2)):
      desc2 = descs2[i_desc2]
      dist = np.linalg.norm(desc2 - desc1)

      if (dist < s_dist):
        s_dist = dist
        s_dist_index = i_desc2
    distances.append(s_dist)

    res_1.append(p1[i_desc1])
    res_2.append(p2[s_dist_index])
    res_d_1.append(desc1)
    res_d_2.append(descs2[s_dist_index])
    res_d_d.append(s_dist)

  distances.sort()
  res_d_d_np = np.array(res_d_d)
  res_1_np = np.array(res_1)
  res_2_np = np.array(res_2)
  res_d_1_np = np.array(res_d_1)
  res_d_2_np = np.array(res_d_2)

  index = np.argsort(res_d_d_np)
  res_1_np = res_1_np[index]
  res_2_np = res_2_np[index]
  res_d_1_np = res_d_1_np[index]
  res_d_2_np = res_d_2_np[index]
  res_d_d_np = res_d_d_np[index]

  mean_dist = np.mean(distances[:10]) * 5
  counter = 0
  #print("res_d_d_np", res_d_d_np[0])
  #print("mean_dist", mean_dist)
  while (res_d_d_np[counter] < mean_dist and (counter < (len(descs1) - 1))):
    counter += 1

  print(counter)
  print("mean of 10 best times 5 = ", mean_dist)
  if (counter + 1 > len(descs1)):
    print("the end, not included dist:", res_d_d_np[counter + 1])
  else: 
    print("all values are within mean[:10] * 5")

  print("this is counter val", counter)
  print("this is len of res-1-np", len(res_1_np))
  res_1_np = res_1_np[:counter]
  res_2_np = res_2_np[:counter]
  res_d_1_np = res_d_1_np[:counter]
  res_d_2_np = res_d_2_np[:counter]

  return(res_1_np, res_2_np, res_d_1_np, res_d_2_np)


def advanced_oneNN(descss1, descss2, pp1, pp2):
  print("calculating advanced oneNN")
  (res_p1, res_p2, res_des1, res_des2) = oneNN(descss1, descss2, pp1, pp2)
  new_res_p1 = []
  new_res_p2 = []


  for point1 in range(0, len(res_p1)):
    fst_shortest_index = point1
    fst_shortest_dist = np.linalg.norm(res_des1[point1] - res_des2[point1])
    scn_shortest_index = 0
    scn_shortest_dist = float("inf")
    
    for point2 in range(0, len(pp2)):
      if (np.all(descss2[point2] != res_des2[point1])):
        dist = np.linalg.norm(res_des1[point1] - descss2[point2])
        #print(dist, fst_shortest_dist)
        if (scn_shortest_dist > dist):
          scn_shortest_dist = dist
          scn_shortest_index = point2

      #else:
      #  print(point2, "these are identical")

    if (fst_shortest_dist / scn_shortest_dist > 0.2):
      #print(fst_shortest_dist/ scn_shortest_dist)
      new_res_p1.append(res_p1[point1])
      new_res_p2.append(res_p2[point1])

  """
  I = cv2.imread("mark-seg-2-1.jpg")
  I2 = cv2.imread("mark-seg-2-2.jpg")
  print("length of the adv_oneNN finall points ", len(new_res_p1))
  #np.save("new_res_p1", new_res_p1)
  #np.save("new_res_p2", new_res_p2)

  cv2.imwrite("zzonennadv" + "one" + ".jpg", I)
  cv2.imwrite("zzonennadv" + "two" + ".jpg", I2)
  """

  print("returned")
  return(new_res_p1, new_res_p2)

def transform_coordinates(f_list):
  new_f = []
  i = 1
  for ele in f_list:
    point = [ele[0][0], ele[0][1], ele[0][2], ele[0][3]]
    desc = ele[1]
    if (ele[0][2] == 6 or ele[0][2] == 7):
      point = [ele[0][0] * 2, ele[0][1] * 2, ele[0][2], ele[0][3]]
    if (ele[0][2] == 11 or ele[0][2] == 12):
      point = [ele[0][0] * 4, ele[0][1] * 4, ele[0][2], ele[0][3]]
    if (ele[0][2] == 16 or ele[0][2] == 17):
      point = [ele[0][0] * 8, ele[0][1] * 8, ele[0][2], ele[0][3]]
    new_f.append([list(point), np.array(desc)])
  new_f = np.array(new_f)
  return(new_f)


def orientation(p):
  """
  Input : Image I, point point
  Output: orientation of image
  """
  x = p[0]
  y = p[1]
  theta = 0.5 * math.pi - math.atan2(x, y)
  theta = math.degrees(theta % (2 * math.pi))
  #print(theta)
  return(theta)

def circle_matrix(mat):
  n = len(mat)
  a = b = n/2
  r = a - 1
  y,x = np.ogrid[-a:n-a, -b:n-b]
  mask = x*x + y*y <= r*r

  array = np.zeros((n,n))
  array[mask] = mat[mask]
  return(array)


def remove_bad_matches(p1, p2):
  dist_yx = float("inf")
  diff_yx = p1[:,:2] - p2[:,:2]
  before = dist_yx
  pic_inc = 0
  while (dist_yx > 100):
    plt.clf()
    mean_yx = np.mean(diff_yx, axis=0)
    #mean_yx = diff_yx[0]
    std_dev_yx = np.std(diff_yx, axis=0)
    dist_yx = math.sqrt((std_dev_yx[0]) ** 2 + (std_dev_yx[1]) ** 2)

    points1 = []
    points2 = []
    del_arr = []
    for i in range(0, len(diff_yx)):
      if (math.sqrt((diff_yx[i, 0] - mean_yx[0]) ** 2 + (diff_yx[i, 1] - mean_yx[1]) ** 2) > dist_yx):
        del_arr.append(i)
    diff_yx = np.delete(diff_yx, del_arr, axis=0)
    p1 = np.delete(p1, del_arr, axis=0)
    p2 = np.delete(p2, del_arr, axis=0)
    #print(len(diff_yx))
    #print(dist_yx)

    plt.scatter(diff_yx[:,0], diff_yx[:,1])
    circle = plt.Circle((mean_yx[0], mean_yx[1]), radius=dist_yx, color="red", fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle)
    mid = plt.scatter(mean_yx[0], mean_yx[1], color='red')
    #plt.show()

    p1 = np.array(p1)
    p2 = np.array(p2)
    pic_inc += 1
    plt.savefig("median_5000_points" + str(pic_inc) + ".jpg")

  return(p1, p2)

def points_ok(pp):
  """
  input: npy points
  output: gives stats of points belonging to Iname, and next picture, ie.
  if input = IMG_9370, then it will return stats for IMG_9370, and IMG_9371
  """
  points_not_matched_1 = np.zeros(20)
  for i in range(0, len(pp)):
    points_not_matched_1[pp[i, 2]] += 1
  return(points_not_matched_1)

def reshapeArr(dd):
  new = np.array(dd)
  for i in range(0, len(dd)):
    new[i,1] = dd[i,1].flatten()
  return new
