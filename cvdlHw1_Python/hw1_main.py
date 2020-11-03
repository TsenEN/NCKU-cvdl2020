# -*- coding: utf-8 -*-

import sys
from hw1_ui import Ui_MainWindow
import cv2 as cv
import cv2
import numpy as np
import glob
import os
from PyQt5.QtWidgets import QMainWindow, QApplication
from matplotlib import pyplot as plt
cont4 = 0
machingKpoint1 = 0
machingKpoint2 = 0
img4_match = 0
list_keypoints1 = []
list_keypoints2 = []

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)

    def on_btn1_1_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images= glob.glob('../images/CameraCalibration/*.bmp')

        i=0
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            i=i+1
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv.drawChessboardCorners(img, (11,8), corners2,ret)

                cv.namedWindow(str(i),cv.WINDOW_GUI_NORMAL )
                cv.imshow(str(i),img)
                cv.waitKey(500)

        cv.destroyAllWindows()

    def on_btn1_2_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images= glob.glob('../images/CameraCalibration/*.bmp')

        i=0
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            i=i+1
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print(mtx)
        cv.waitKey(500)
        cv.destroyAllWindows()

    def on_btn1_3_click(self):
        # get the input from ui item
        number = int(self.cboxImgNum.currentText())

         # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        # read images
        path = '../images/CameraCalibration/'+ str(number)+'.bmp'
        img = cv.imread(path)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        print(path)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (11,8),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

            # get rotation matrix and plus tranalation matrix
            R, jacobian = cv.Rodrigues(rvecs[0])
            extrinsic = np.hstack((R,tvecs[0]))
            print(extrinsic)


    def on_btn1_4_click(self):
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images = glob.glob('../images/CameraCalibration/*.bmp')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        print(dist)


    def on_btn2_1_click(self):
        # define criteria = (type,max_iter,epsilon)
        # cv.TERM_CRITERIA_EPS :The accuracy (error) meets the epsilon stop , accuracy = 0.001
        # cv.TERM_CRITERIA_MAX_ITER：The number of iterations exceeds max_iter stop ,max_iter = 30.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        #read all the images from folder
        images = glob.glob('../images/CameraCalibration/*.bmp')

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        # Function to draw the axis
        def draw(img, corners, imgpts):
            imgpts = np.int32(imgpts).reshape(-1,2) 
            # draw ground floor in green(0,255,0)
            img = cv.drawContours(img, [imgpts[:4]],-1,(0,0,255),10)

            # draw pillars in blue color
            for i,j in zip(range(4),range(4,8)):
                img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),10)

            # draw top layer in red(0,0,255) color
            # img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),10)
            return img

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((8*11,3), np.float32)
        objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
        axis = np.float32([[1,1,0], [3,5,0], [3,5,0], [5,1,0],[3,3,-3],[3,3,-3],[3,3,-3],[3,3,-3] ])

        # declare a array to store video frame
        Video_img=[]
        for fname in glob.glob('../images/Augment/*.bmp'):
            img = cv.imread(fname)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11,8),None)

            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                _,rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners2, mtx, dist)

                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

                img = draw(img,corners2,imgpts)
                Video_img.append(img)
                cv.imshow('Video',img)
                cv.waitKey(500)

        # making vidoe
        height,width,layers=Video_img[1].shape
        video=cv.VideoWriter('video.mp4',-1,2,(width,height))
        for j in range(0,5):
            video.write(Video_img[j])

        cv.destroyAllWindows()


    def on_btn3_1_click(self):
        # read left and right images
        imgL = cv.imread('../images/imL.png',0)
        imgR = cv.imread('../images/imR.png',0)

        # making disparity map
        stereo = cv.StereoSGBM_create(numDisparities=16, blockSize=20) #the third parameter
        disparity = stereo.compute(imgL,imgR)

        # normalization
        normalized_img = np.zeros((800, 800))
        normalized_img = cv.normalize(disparity, normalized_img, 0, 255, cv.NORM_MINMAX,cv.CV_8U)

        cv.imshow('Without L-R Disparity Check',normalized_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    

    def on_btn4_1_click(self):
        global cont4
        if  cont4 == 0 : 
            cont4 = cont4 | 1      
            Aerial1 = cv.imread('../images/Aerial1.jpg')
            Aerial2 = cv.imread('../images/Aerial2.jpg')
            gray1= cv.cvtColor(Aerial1,cv.COLOR_BGR2GRAY)
            gray2= cv.cvtColor(Aerial2,cv.COLOR_BGR2GRAY)
            # construct a SIFT object
            sift1 = cv.xfeatures2d.SIFT_create()
            sift2 = cv.xfeatures2d.SIFT_create()
            # finds the keypoint
            keypoints_1, descriptors_1 = sift1.detectAndCompute(gray1,None)   
            keypoints_2, descriptors_2 = sift2.detectAndCompute(gray2,None)

            # BFMatcher with default params
            bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
            matches = bf.match(descriptors_1,descriptors_2)
            matches =sorted(matches, key=lambda x:x.distance)
            global img4_match 
            img4_match = cv.drawMatches(gray1, keypoints_1, gray2, keypoints_2, matches[:6], gray2,(0,255,0) ,flags=2)

            i = 0
            while i < 6:
            # Get the matching keypoints for each of the images
                img1_index = matches[i].queryIdx
                img2_index = matches[i].trainIdx

                list_keypoints1.append(keypoints_1[img1_index])
                list_keypoints2.append(keypoints_2[img2_index])

                i += 1
            # save the image
            global machingKpoint1
            machingKpoint1 = cv.drawKeypoints(gray1, list_keypoints1, machingKpoint1)
            cv.imwrite('FeatureAerial1.jpg', machingKpoint1)
            global machingKpoint2
            machingKpoint2 = cv.drawKeypoints(gray2, list_keypoints2, machingKpoint2)
            cv.imwrite('FeatureAerial2.jpg', machingKpoint2)
            # show the result
            cv.imshow('result1',np.hstack((machingKpoint1,machingKpoint2)))
            cv.waitKey(0)
            cv.destroyAllWindows()
        else :
            cv.imshow('result1',np.hstack((machingKpoint1,machingKpoint2)))
            cv.waitKey(0)
            cv.destroyAllWindows()

    def on_btn4_2_click(self):
        global cont4
        if  cont4 == 0 : 
            print('Please generate 4.1 keypoints first')
        else :
            cv.imshow('result2',img4_match)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
