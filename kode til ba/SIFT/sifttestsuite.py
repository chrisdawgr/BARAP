import siftdetector as sdet
import math
import helperfunctions as h
import siftorientation as ori 
import siftdescriptor as sdes
import numpy as np
import cv2


def sifttestsuite(pic1, pic2):
  fn1 = pic1[:-4]
  fn2 = pic2[:-4]

  p1 = sdet.SIFT(pic1, 1.2)
  p2 = sdet.SIFT(pic2, 1.2)
  np.save(fn1, p1)
  np.save(fn2, p2)

  p1 = np.load(fn1 + ".npy")
  p2 = np.load(fn2 + ".npy")
  print(len(p1))
  print(len(p2))

  p1 = ori.sift_orientation(pic1, p1)
  p2 = ori.sift_orientation(pic2, p2)

  np.save(fn1, p1)
  np.save(fn2, p2)

  print("hier")
  p1 = np.load(fn1 + ".npy")
  p2 = np.load(fn2 + ".npy")

  final_desc1 = sdes.sift_descriptor(pic1, p1)
  final_desc2 = sdes.sift_descriptor(pic2, p2)
  np.save(fn1 + "-desc", final_desc1)
  np.save(fn2 + "-desc", final_desc2)
  f1 = np.load(fn1 + "-desc.npy")
  f2 = np.load(fn2 + "-desc.npy")
  
  # MUST 
  f1 = h.transform_coordinates(f1)
  f2 = h.transform_coordinates(f2)
  print(len(f1))
  print(len(f2))
  np.save(fn1 + "-desc-trans", f1)
  np.save(fn2 + "-desc-trans", f2)
  f1 = np.load(fn1 + "-desc-trans.npy")
  f2 = np.load(fn2 + "-desc-trans.npy")

  f1 = np.load(fn1 + "-desc-trans.npy")
  f2 = np.load(fn2 + "-desc-trans.npy")

  #oneNN_wdist nesecary for matching as described in rapport
  (p1, p2, desc1, desc2) = h.oneNN_wdist(f1[:,1], f2[:,1], f1[:,0], f2[:,0])
  h.drawMatches2(pic1, p1[:32], pic2, p2[:32], [])
  np.save(fn1 + "onenn-p", [p1, p2])
  np.save(fn1 + "onenn-d", [desc1, desc2])

def removematches(pic1, pic2, sift_or_surf):
  fn1 = pic1[:-4]
  px = np.load(fn1 + "onenn-p.npy")
  p1 = px[0]
  p2 = px[1]
  print("this is len p1:",len(p1))
  half_p = len(p1) / 4
  (p1, p2) = h.remove_bad_matches(p1, p2)
  matches1 = h.points_ok(p1, sift_or_surf)
  matches2 = h.points_ok(p2, sift_or_surf)
  h.drawMatches2(pic1, p1, pic2, p2, [])
  #h.drawMatches2(pic1, p1[:15], pic2, p2[:15], [])
  print("Number of matches: ", len(p1))
  with open("repeatabilitymeasure.txt", "a") as file:
    det1 = len(np.load(pic1[:-4] + ".npy"))
    det2 = len(np.load(pic2[:-4] + ".npy"))
    mean = (det1 + det2) / 2.0
    picn1 = "IMG" + "\\_" + pic1[4:-4] + ".jpg"
    picn2 = "IMG" + "\\_" + pic2[4:-4] + ".jpg"
    rep_mes = len(p1) / (mean * 1.0)
    file.write(picn1 + " &\t" + picn2 + " &\t" + str(det1) + " &\t" + str(det2) + " &\t" + str(mean) + \
    " &\t" + str(len(p1)) + " &\t" + str(rep_mes) + "\\\\ \hline\n")
  return(matches1, matches2)

def print_diffs(pic1, pic2, sift_or_surf):
  matches_1, matches_2 = removematches(pic1, pic2, sift_or_surf)
  orig_1 = h.points_ok(np.load(pic1[:-4] + ".npy"), sift_or_surf)
  orig_2 = h.points_ok(np.load(pic2[:-4] + ".npy"), sift_or_surf)
  print(matches_1)
  #print(orig_1)
  #print(orig_2)
  #print(matches_1)
  #print(matches_2)
  with open("detector-matches.txt", "a") as file:
    file.write( \
        pic1 + "\t"
        + str(orig_1[11]) + "\t" + str(orig_1[12]) + "\t" \
        + str(orig_1[16]) + "\t" + str(orig_1[17]) + "\t" \
        + str(matches_1[11]) + "\t" + str(matches_1[12]) + "\t" \
        + str(matches_1[16]) + "\t\t" + str(matches_1[17]) + "\n")

    file.write( \
        pic2 + "\t"
        + str(orig_2[11]) + "\t" + str(orig_2[12]) + "\t" \
        + str(orig_2[16]) + "\t" + str(orig_2[17]) + "\t" \
        + str(matches_2[11]) + "\t" + str(matches_2[12]) + "\t\t" \
        + str(matches_2[16]) + "\t\t" + str(matches_2[17]) + "\n\n")

sifttestsuite("IMG_9371.jpg", "IMG_9372.jpg")
removematches("IMG_9371.jpg", "IMG_9372.jpg",0)
