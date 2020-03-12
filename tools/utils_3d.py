import numpy as np
import cv2
import csv
import pdb
import math
from tools.utils_calib import Calibration

# color_dict = {'Car':[0.2, 0.2, 0.9],
#             'Van':[0.4, 0.2, 0.4],
#             'Truck':[0.6, 0.2, 0.6],
#             'Pedestrian':[0.9, 0.2, 0.2],
#             'Person_sitting':[0.6, 0.2, 0.4],
#             'Cyclist':[0.2, 0.9, 0.2],
#             'Tram':[0.2, 0.6, 0.2],
#             'Misc':[0.2, 0.4, 0.2],
#             'DontCare':[0.2, 0.2, 0.2]}

color_dict = {'Car':[216, 216, 100],
            'Pedestrian':[0, 210, 0],
            'Cyclist':[0, 128, 255]}

def dottedline(img,p0,p1,linepoints,width,color):
    xpts = np.linspace(p0[0],p1[0],linepoints)
    ypts = np.linspace(p0[1],p1[1],linepoints)
    for xp,yp in zip(xpts,ypts):
        cv2.circle(img,(int(xp),int(yp)),width,color,-1)
    return img

def _draw_projection_obstacle_to_cam(obs, calib_file, bvres, only_front, draw, img=None, bv_img=None, is_kitti_gt=False, n=None, is_kitti_ann=False):
    '''
    This script is used for fullfil two complementary task:
    * Obtain 2D bbox in camera view
    * Draw BV and camera predictions in 3D
    It totally depends on the parameter draw
    '''
    
    calib_c = Calibration(calib_file)

    yaw = -obs.yaw + math.pi/2

    if is_kitti_gt or is_kitti_ann:
        xyz = calib_c.project_rect_to_velo(np.array([[obs.location.x,obs.location.y,obs.location.z]]))
        x = xyz[0][0]
        y = xyz[0][1]
        z = xyz[0][2] + obs.height / 2
    else:
        x = obs.location.x
        y = obs.location.y
        z = obs.location.z + obs.height / 2

    l = obs.length
    h = obs.height
    w = obs.width

    centroid = np.array([x, y])
    
    corners = np.array([
        [x - l / 2., y + w / 2.],
        [x + l / 2., y + w / 2.],
        [x + l / 2., y - w / 2.],
        [x - l / 2., y - w / 2.]
    ])

    # Compute rotation matrix
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])

    # Rotate all corners at once by yaw
    rot_corners = np.dot(corners - centroid, R.T) + centroid

    if draw:
        # Convert x to x in BV
        xs = bv_img.shape[1] / 2 - rot_corners[:, 1] / bvres
        ys = (bv_img.shape[0] if only_front else bv_img.shape[0]/2) - rot_corners[:, 0] / bvres

        xsc = bv_img.shape[1] / 2 - centroid[1] / bvres
        ysc = (bv_img.shape[0] if only_front else bv_img.shape[0]/2) - centroid[0] / bvres

        pt1 = np.array([xs[0], ys[0]])
        pt2 = np.array([xs[1], ys[1]])
        pt3 = np.array([xs[2], ys[2]])
        pt4 = np.array([xs[3], ys[3]])

        ctr = np.array([pt1, pt2, pt3, pt4]).reshape((-1, 1, 2)).astype(np.int32)

        for j in range(4):
            k = (j + 1) % 4
            if is_kitti_gt:
                # print(ctr[j][0],ctr[k][0])
                bv_img = dottedline(bv_img,ctr[j][0],ctr[k][0],5,2,(0,135,135))
            else:
                cv2.line(bv_img,( int(ctr[j][0][0]),   int(ctr[j][0][1])),
                         ( int(ctr[k][0][0]), int(ctr[k][0][1])), 
                         color_dict[obs.kind_name],3)
                arrow_len = (l/bvres)/2.+10
                xs1 = arrow_len*math.cos(obs.yaw)
                ys1 = arrow_len*math.sin(obs.yaw)
                bv_img = cv2.arrowedLine(bv_img, (int(xsc),int(ysc)), (int(xs1+xsc),int(ys1+ysc)), 
                                             color_dict[obs.kind_name],3)

    x1 = rot_corners[0,0]
    x2 = rot_corners[1,0]
    x3 = rot_corners[2,0]
    x4 = rot_corners[3,0]

    y1 = rot_corners[0,1]
    y2 = rot_corners[1,1]
    y3 = rot_corners[2,1]
    y4 = rot_corners[3,1]

    # Project the 8 vertices of the prism
    vertices = []
    vertices.append([x1, y1, z+h/2])
    vertices.append([x2, y2, z+h/2])
    vertices.append([x3, y3, z+h/2])
    vertices.append([x4, y4, z+h/2])
    vertices.append([x1, y1, z-h/2])
    vertices.append([x2, y2, z-h/2])
    vertices.append([x3, y3, z-h/2])
    vertices.append([x4, y4, z-h/2])

    image_pts = calib_c.project_velo_to_image(np.array(vertices))

    #3D draw: front-backs-sides
    if draw:
        if is_kitti_gt:
            for j in np.arange(0,8,2):
                # print(image_pts[j][0],[image_pts[j][0],image_pts[j][1]],[image_pts[j+1][0],image_pts[j+1][1]])
                img = dottedline(img,[image_pts[j][0],image_pts[j][1]],[image_pts[j+1][0],image_pts[j+1][1]],10,2,(0,255,255))
            img = dottedline(img,[image_pts[0][0],image_pts[0][1]],[image_pts[4][0],image_pts[4][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[4][0],image_pts[4][1]],[image_pts[7][0],image_pts[7][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[7][0],image_pts[7][1]],[image_pts[3][0],image_pts[3][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[3][0],image_pts[3][1]],[image_pts[0][0],image_pts[0][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[1][0],image_pts[1][1]],[image_pts[5][0],image_pts[5][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[5][0],image_pts[5][1]],[image_pts[6][0],image_pts[6][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[6][0],image_pts[6][1]],[image_pts[2][0],image_pts[2][1]],7,2,(0,255,255))
            img = dottedline(img,[image_pts[2][0],image_pts[2][1]],[image_pts[1][0],image_pts[1][1]],7,2,(0,255,255))
        else:
            for j in range(0,3):
            #0,0-3,0
                cv2.line(img,( int(np.ceil(image_pts[j][0])),   int(np.ceil(image_pts[j][1]))),
                              ( int(np.ceil(image_pts[j+1][0])), int(np.ceil(image_pts[j+1][1]))), 
                              color_dict[obs.kind_name],3)
                #4,0-7,0
                cv2.line(img,( int(np.ceil(image_pts[j+4][0])),   int(np.ceil(image_pts[j+4][1]))),
                              ( int(np.ceil(image_pts[j+5][0])), int(np.ceil(image_pts[j+5][1]))), 
                              color_dict[obs.kind_name],3)
            cv2.line(img,( int(np.ceil(image_pts[0][0])),   int(np.ceil(image_pts[0][1]))),
                              ( int(np.ceil(image_pts[3][0])), int(np.ceil(image_pts[3][1]))), 
                              color_dict[obs.kind_name],3)
            cv2.line(img,( int(np.ceil(image_pts[4][0])),   int(np.ceil(image_pts[4][1]))),
                              ( int(np.ceil(image_pts[7][0])), int(np.ceil(image_pts[7][1]))), 
                              color_dict[obs.kind_name],3)
            for j in range(0,2):
                cv2.line(img,( int(np.ceil(image_pts[j*2][0])),   int(np.ceil(image_pts[j*2][1]))),
                              ( int(np.ceil(image_pts[j*2+4][0])), int(np.ceil(image_pts[j*2+4][1]))), 
                              color_dict[obs.kind_name],3)
                cv2.line(img,( int(np.ceil(image_pts[j*2+1][0])),   int(np.ceil(image_pts[j*2+1][1]))),
                              ( int(np.ceil(image_pts[j*2+5][0])), int(np.ceil(image_pts[j*2+5][1]))), 
                              color_dict[obs.kind_name],3)

    # Extreme object points in the image
    image_u1 = np.min(image_pts[:, 0]) # Limits for kitti dataset
    image_v1 = np.min(image_pts[:, 1])

    image_u2 = np.max(image_pts[:, 0])
    image_v2 = np.max(image_pts[:, 1])
    if (image_u1 <= 0 and image_u2 <= 0) or \
        (image_u1 >= 1242. - 1 and image_u2 >= 1242. - 1) or \
        (image_v1 <= 0 and image_v2 <= 0) or \
        (image_v1 >= 375. - 1 and image_v2 >= 375. - 1):
            return img, bv_img, None
    image_u1 = np.min((np.max((image_u1,0.)),1242.))
    image_v1 = np.min((np.max((image_v1,0.)),375.))
    image_u2 = np.min((np.max((image_u2,0.)),1242.))
    image_v2 = np.min((np.max((image_v2,0.)),375.))

    # if draw:
    #     #2D draw: front-side
    #     if is_kitti_gt:
    #         cv2.rectangle(img, (int(image_u1), int(image_v1)), 
    #                     (int(image_u2), int(image_v2)), 
    #                     (255,255,255), 2)
    #     else:
    #         cv2.rectangle(img, (int(image_u1), int(image_v1)), 
    #                     (int(image_u2), int(image_v2)), 
    #                     (100,100,100), 4)

    return img, bv_img, [image_u1, image_v1, image_u2, image_v2]
