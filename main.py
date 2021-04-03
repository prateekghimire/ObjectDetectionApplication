from PyQt5.QtWidgets import QMainWindow,QApplication,QPushButton,QProgressBar,QLabel,QVBoxLayout,QHBoxLayout,QGroupBox,QDialog,QGridLayout
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import sys
from imageai import Detection
import cv2
import imutils

modelpath="models/yolo.h5"
yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel()

class Window(QMainWindow,QLabel):
    def __init__(self):
        super().__init__()
        self.title="OBJECT DETECTION APP"
        self.icon="image/ico.png"
        self.InitWindow()
        

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setWindowIcon(QtGui.QIcon(self.icon))
        self.setStyleSheet("QMainWindow{background-image:url(image/bg.jpg)}")
        self.setFixedSize(1024,760)
        self.UIcomponents()
        self.show()

    def UIcomponents(self):
        startbtn=QPushButton("TEST VIDEO",self)
        startbtn.move(450,260)
        startbtn.setFixedSize(150,50)
        startbtn.setIcon(QtGui.QIcon("image/video.ico"))
        startbtn.setStyleSheet("QPushButton{color:white;background-color:brown;border-radius:25px;border: 8px solid white;font: bold}"
        "QPushButton:pressed{color:black;background-color:green}")
        startbtn.clicked.connect(self.start)

        recordbtn=QPushButton("TEST WEBCAM",self)
        recordbtn.move(450,360)
        recordbtn.setFixedSize(150,50)
        recordbtn.setIcon(QtGui.QIcon("image/camera.ico"))
        recordbtn.setStyleSheet("QPushButton{color:white;background-color:brown;border-radius:25px;border: 8px solid white;font: bold}"
        "QPushButton:pressed{color:black;background-color:green}")
        recordbtn.clicked.connect(self.webcam)

        exitbtn=QPushButton("EXIT",self)
        exitbtn.move(450,460)
        exitbtn.setFixedSize(150,50)
        exitbtn.clicked.connect(self.close)
        exitbtn.setIcon(QtGui.QIcon("image/exit.png"))
        exitbtn.setStyleSheet("QPushButton{color:white;background-color:brown;border-radius:25px;border: 8px solid white;font: bold}"
        "QPushButton:pressed{color:black;background-color:red}")

    def start(self):
        video=cv2.VideoCapture("video/video.avi")  #Give direction of test video
        while video.isOpened():
            ret,frame=video.read()
            if ret==True:
                frame=imutils.resize(frame,width=800,height=800)
                img, preds = yolo.detectCustomObjectsFromImage(input_image=frame, 
                      custom_objects=None, input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=False,
                      display_object_name=True)
                cv2.imshow("", img) 
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
            else:
                break
        video.release()
        cv2.destroyAllWindows()
        
    def close(self):
        sys.exit()

    def webcam(self):
        cam = cv2.VideoCapture(1)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

        while True:
            ret, img = cam.read()
            img, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=None, input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=False,
                      display_object_name=True)
            cv2.imshow("", img)    
            if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
                break

        cam.release()
        cv2.destroyAllWindows()

App=QApplication(sys.argv)
window=Window()
sys.exit(App.exec())
