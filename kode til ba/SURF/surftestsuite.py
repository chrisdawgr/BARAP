import surfdescriptor as sdes
import usurfdescriptor as udes
import surfdetector as sdet
import helperfunctions as h
import surforientation as ori
import cv2
import numpy as np
import os

def surf_tester(pic1, pic2, rot):
  print ("testing with", pic1, pic2)
  name1 = pic1[:-4]
  name2 = pic2[:-4]

  p1 = sdet.findSurfPoints(pic1)
  p2 = sdet.findSurfPoints(pic2)
  np.save(name1, p1)
  np.save(name2, p2)
  
  p1 = np.load(name1 + ".npy")
  p2 = np.load(name2 + ".npy")

  if rot == 1:
    p1 = np.load(name1 + ".npy")
    p2 = np.load(name2 + ".npy")
    p1 = ori.surf_orientation(pic1, p1)
    p2 = ori.surf_orientation(pic2, p2)
    desc1po1 = sdes.surf_descriptor_w_orientation(pic1, p1)
    desc2po2 = sdes.surf_descriptor_w_orientation(pic2, p2)
  if rot == 0:
    p1 = sdet.findSurfPoints(pic1)
    p2 = sdet.findSurfPoints(pic2)
    np.save(name1, p1)
    np.save(name2, p2)
    desc1po1 = udes.surf_descriptor(pic1, p1)
    desc2po2 = udes.surf_descriptor(pic2, p2)
  np.save(name1 + "-desc", desc1po1)
  np.save(name2 + "-desc", desc2po2)
  #oneNN (des1, des2, p1, p2)
  
  desc1po1 = np.load(name1 + "-desc.npy")
  desc2po2 = np.load(name2 + "-desc.npy")

  newdesc1po1 = []
  newdesc2po2 = []

  for i in desc1po1:
    if i[0][2] == 6.8 or i[0][2] == 10.0:
      newdesc1po1.append(i) 
  for i in desc2po2:
    if i[0][2] == 6.8 or i[0][2] == 10.0:
      newdesc2po2.append(i)
  
  newdesc2po2 = np.array(newdesc2po2)
  newdesc1po1 = np.array(newdesc1po1)
  """
  desc1po1 = np.load(name1 + "-desc.npy")
  desc2po2 = np.load(name2 + "-desc.npy")

#IMPORTTANT
  desc1po1 = h.reshapeArr(desc1po1)
  desc2po2 = h.reshapeArr(desc2po2)

  (sp1, sp2, d1, d2) = h.oneNN_wdist(desc1po1[:,1], desc2po2[:,1], desc1po1[:,0], desc2po2[:,0])
  np.save(name1 + "onenn-p", [sp1, sp2])
  np.save(name2 + "onenn-d", [d1, d2])
  sp1, sp2 = h.remove_bad_matches(sp1, sp2)
  #(sp1, sp2) = h.advanced_oneNN(desc1po1[:,1], desc2po2[:,1], desc1po1[:,0], desc2po2[:,0])
  #print(sp1[0:10])
  #print("\n")
  #print(sp2[0:10])

  h.drawMatches2(pic1, sp1, pic2, sp2, [])

  """
  #      test for drawMatches   
  desc1po1 = np.load("room10-desc.npy")
  desc2po2 = np.load("room11-desc.npy")

  desc1po1 = desc1po1[np.argsort(desc1po1[:,0])]
  desc2po2 = desc2po2[np.argsort(desc2po2[:,0])]

  desc1po1 = [desc1po1[523,0]]
  desc2po2 = [desc2po2[327,0]]

  desc1po1.append([0,0,0])
  desc2po2.append([0,0,0])
  h.drawMatches(pic1, desc1po1, pic2, desc2po2, [])
  """
surf_tester("IMG_9370.jpg", "IMG_9371.jpg",0)
surf_tester("IMG_9371.jpg", "IMG_9372.jpg",0)
surf_tester("IMG_9372.jpg", "IMG_9373.jpg",0)
surf_tester("IMG_9373.jpg", "IMG_9374.jpg",0)
surf_tester("IMG_9374.jpg", "IMG_9375.jpg",0)
surf_tester("IMG_9375.jpg", "IMG_9376.jpg",0)
surf_tester("IMG_9376.jpg", "IMG_9377.jpg",0)
surf_tester("IMG_9377.jpg", "IMG_9378.jpg",0)
surf_tester("IMG_9378.jpg", "IMG_9379.jpg",0)
surf_tester("IMG_9379.jpg", "IMG_9380.jpg",0)
