import face_alignment
import cv2
import numpy as np
from glob import glob
from pathlib import PurePath, Path
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip

import torch
print (torch.cuda.is_available())
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False) # device='cpu'

def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(1)
      #cv2.destroyAllWindows()

vp_clip = VideoFileClip("old_version/vm.mp4")
for i in range(20):
    face = vp_clip.get_frame(i)
    #imgs(x)
    preds = fa.get_landmarks(face)
    
    A_x = None
    A_y = None
    
    if preds is not None:
        preds = preds[0]
#        mask = np.zeros_like(x)
        mask = np.zeros((face.shape[0], face.shape[1]), np.uint8)        
        
        # Draw mouth binary mask
#        pnts_mouth = [(preds[i,0], preds[i,1]) for i in range(48,60)]
        pnts_mouth = [(preds[i,0], preds[i,1]) for i in range(0,68)]
        points = np.array(pnts_mouth, np.int32)
        #print (pnts_mouth)
        
        # Center
#        pnts_start, pnts_end = pnts_mouth[0], pnts_mouth[-1]
#        center_of_forehead = ((pnts_start[0] + pnts_end[0]) // 2, (pnts_start[1] + pnts_end[1]) // 2)
#        print (center_of_forehead)
        
        hull = cv2.convexHull(points)
        face_cp = face.copy()
        #imgs(cv2.cvtColor((cv2.polylines(face_cp, [hull], True, (255,255,255), 1)), cv2.COLOR_BGR2RGB))
        
        face_image_1 = cv2.bitwise_and(face, face, mask=mask)
        
        
        rect = cv2.boundingRect(hull)
        #print (rect)
        subdiv = cv2.Subdiv2D(rect) # Creates an instance of Subdiv2D
        for p in pnts_mouth:
            #print (p)
            subdiv.insert(p) # Insert points into subdiv
            
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        face_cp = face.copy()

        def get_index(arr):
            index = 0
            if arr[0].any():
                index = arr[0][0]
            return index

        for triangle in triangles :

            # Gets the vertex of the triangle
            pt1 = (triangle[0], triangle[1])
            pt2 = (triangle[2], triangle[3])
            pt3 = (triangle[4], triangle[5])
            
            # Draws a line for each side of the triangle
            cv2.line(face_cp, pt1, pt2, (255, 255, 255), 1,  0)
            cv2.line(face_cp, pt2, pt3, (255, 255, 255), 1,  0)
            cv2.line(face_cp, pt3, pt1, (255, 255, 255), 1,  0)

            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = get_index(index_pt1)
            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = get_index(index_pt2)
            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = get_index(index_pt3)

            # Saves coordinates if the triangle exists and has 3 vertices
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                vertices = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(vertices)

        imgs(cv2.cvtColor(face_cp, cv2.COLOR_BGR2RGB))
        
# ------------------------------------->    
#    
 
    else:
        mask = np.zeros_like(y)
        print(f"No faces were detected in image ")   

#Текст:
#Это тестовая презентация, с использованием видеокарты потребительского сегмента, 980ti,
