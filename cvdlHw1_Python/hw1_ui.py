# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\hw1.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1082, 422)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # 1. Calibration
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")

        # 2. Augmented Reality
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(400, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        # 3. Stereo Disparity Map
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(400, 100, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        # 4. SIFT
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(400, 180, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        # 5. Training Cifar10 Classifier Using VGG16
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(700, 20, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")  
        

        # layout of 1
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(50, 50, 341, 260))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame.setObjectName("frame")


        self.btn1_1 = QtWidgets.QPushButton(self.frame)
        self.btn1_1.setGeometry(QtCore.QRect(10, 50, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_1.setFont(font)
        self.btn1_1.setObjectName("btn1_1")

        self.btn1_2 = QtWidgets.QPushButton(self.frame)
        self.btn1_2.setGeometry(QtCore.QRect(10, 110, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_2.setFont(font)
        self.btn1_2.setObjectName("btn1_2")

        self.btn1_4 = QtWidgets.QPushButton(self.frame)
        self.btn1_4.setGeometry(QtCore.QRect(10, 170, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_4.setFont(font)
        self.btn1_4.setObjectName("btn1_4")

        # layout of 1.3
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(160, 70, 171, 141))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.frame_2.setFont(font)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_2.setObjectName("frame_2")

        self.btn1_3 = QtWidgets.QPushButton(self.frame_2)
        self.btn1_3.setGeometry(QtCore.QRect(30, 90, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn1_3.setFont(font)
        self.btn1_3.setObjectName("btn1_3")

        # 1.3 select image
        self.label_1_3_S = QtWidgets.QLabel(self.frame_2)
        self.label_1_3_S.setGeometry(QtCore.QRect(10, 10, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_1_3_S.setFont(font)
        self.label_1_3_S.setObjectName("label_1_3_S")

        # 1.3 Drop-down menu
        self.cboxImgNum = QtWidgets.QComboBox(self.frame_2)
        self.cboxImgNum.setGeometry(QtCore.QRect(10, 50, 151, 31))
        self.cboxImgNum.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cboxImgNum.setObjectName("cboxImgNum")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")
        self.cboxImgNum.addItem("")

        # 1.3 Extrinsic button
        self.label_1_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_1_3.setGeometry(QtCore.QRect(200, 90, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.label_1_3.setFont(font)
        self.label_1_3.setObjectName("label_1_3")

        # layout of 2 
        self.frame_3 = QtWidgets.QFrame(self.centralwidget)
        self.frame_3.setGeometry(QtCore.QRect(410, 50, 261, 50))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_3.setObjectName("frame_3")

        # layout of button2.1
        self.btn2_1 = QtWidgets.QPushButton(self.frame_3)
        self.btn2_1.setGeometry(QtCore.QRect(25, 5, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn2_1.setFont(font)
        self.btn2_1.setObjectName("btn2_1")

        # layout of 3
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(410, 130, 261, 50))
        self.frame_4.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_4.setObjectName("frame_4")

        ## layout of button3.1
        self.btn3_1 = QtWidgets.QPushButton(self.frame_4)
        self.btn3_1.setGeometry(QtCore.QRect(25, 5, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn3_1.setFont(font)
        self.btn3_1.setObjectName("btn3_1")

        # layout of 4 
        self.frame_5 = QtWidgets.QFrame(self.centralwidget)
        self.frame_5.setGeometry(QtCore.QRect(410, 210, 261, 100))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_5.setObjectName("frame_5")

        # layout of button4.1
        self.btn4_1 = QtWidgets.QPushButton(self.frame_5)
        self.btn4_1.setGeometry(QtCore.QRect(25, 5, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_1.setFont(font)
        self.btn4_1.setObjectName("btn4_1")

        # layout of button4.2
        self.btn4_2 = QtWidgets.QPushButton(self.frame_5)
        self.btn4_2.setGeometry(QtCore.QRect(25, 50, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn4_2.setFont(font)
        self.btn4_2.setObjectName("btn4_2")

        # layout of 5 
        self.frame_6 = QtWidgets.QFrame(self.centralwidget)
        self.frame_6.setGeometry(QtCore.QRect(700, 50, 271, 300))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frame_6.setObjectName("frame_6")

        # layout of button5.1
        self.btn5_1 = QtWidgets.QPushButton(self.frame_6)
        self.btn5_1.setGeometry(QtCore.QRect(25, 5, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_1.setFont(font)
        self.btn5_1.setObjectName("btn5_1")

        # layout of button5.2
        self.btn5_2 = QtWidgets.QPushButton(self.frame_6)
        self.btn5_2.setGeometry(QtCore.QRect(25, 50, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_2.setFont(font)
        self.btn5_2.setObjectName("btn5_2")

        # layout of button5.3
        self.btn5_3 = QtWidgets.QPushButton(self.frame_6)
        self.btn5_3.setGeometry(QtCore.QRect(25, 95, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_3.setFont(font)
        self.btn5_3.setObjectName("btn5_3")

        # layout of button5.4
        self.btn5_4 = QtWidgets.QPushButton(self.frame_6)
        self.btn5_4.setGeometry(QtCore.QRect(25, 140, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_4.setFont(font)
        self.btn5_4.setObjectName("btn5_4")

        # layout of button5.5
        self.btn5_5 = QtWidgets.QPushButton(self.frame_6)
        self.btn5_5.setGeometry(QtCore.QRect(25, 225, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.btn5_5.setFont(font)
        self.btn5_5.setObjectName("btn5_5")

        # layout of SpinBox
        self.SpinBox5 = QtWidgets.QSpinBox(self.frame_6)
        self.SpinBox5.setGeometry(QtCore.QRect(25, 185, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.SpinBox5.setFont(font)
        self.SpinBox5.setRange(0, 9999)
        self.SpinBox5.setObjectName("SpinBox5")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 682, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn2_1.setText(_translate("MainWindow", "2.1 Augmented Reality"))
        self.btn3_1.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))
        self.btn4_1.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.btn4_2.setText(_translate("MainWindow", "4.2 Matched keypoints"))
        self.btn5_1.setText(_translate("MainWindow", "5.1 Show Train Image"))
        self.btn5_2.setText(_translate("MainWindow", "5.2 Show hyperparameters"))
        self.btn5_3.setText(_translate("MainWindow", "5.3 Show Model Structure"))
        self.btn5_4.setText(_translate("MainWindow", "5.4 Show Accuracy"))
        self.btn5_5.setText(_translate("MainWindow", "5.5 Test"))
        self.label.setText(_translate("MainWindow", "1. Calibration"))
        self.label_2.setText(_translate("MainWindow", "2. Augmented Reality"))
        self.label_3.setText(_translate("MainWindow", "3. Stereo Disparity Map"))
        self.label_4.setText(_translate("MainWindow", "4. SIFT"))
        self.label_5.setText(_translate("MainWindow", "5. Cifar10 Classifier"))
        self.btn1_2.setText(_translate("MainWindow", "1.2 Intrinsic"))
        self.btn1_1.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.btn1_4.setText(_translate("MainWindow", "1.4 Distortion"))
        self.btn1_3.setText(_translate("MainWindow", "1.3 Extrinsic"))
        self.label_1_3_S.setText(_translate("MainWindow", "Select Image"))
        self.cboxImgNum.setItemText(0, _translate("MainWindow", "1"))
        self.cboxImgNum.setItemText(1, _translate("MainWindow", "2"))
        self.cboxImgNum.setItemText(2, _translate("MainWindow", "3"))
        self.cboxImgNum.setItemText(3, _translate("MainWindow", "4"))
        self.cboxImgNum.setItemText(4, _translate("MainWindow", "5"))
        self.cboxImgNum.setItemText(5, _translate("MainWindow", "6"))
        self.cboxImgNum.setItemText(6, _translate("MainWindow", "7"))
        self.cboxImgNum.setItemText(7, _translate("MainWindow", "8"))
        self.cboxImgNum.setItemText(8, _translate("MainWindow", "9"))
        self.cboxImgNum.setItemText(9, _translate("MainWindow", "10"))
        self.cboxImgNum.setItemText(10, _translate("MainWindow", "11"))
        self.cboxImgNum.setItemText(11, _translate("MainWindow", "12"))
        self.cboxImgNum.setItemText(12, _translate("MainWindow", "13"))
        self.cboxImgNum.setItemText(13, _translate("MainWindow", "14"))
        self.cboxImgNum.setItemText(14, _translate("MainWindow", "15"))
        self.label_1_3.setText(_translate("MainWindow", "1.3 Extrinsic"))
