import sys
sys.path.append("/usr/local/lib/python3.9/site-packages")
import cv2
from dronekit import *
from pymavlink import mavutil
import time
import numpy as np

master = mavutil.mavlink_connection('/dev/ttyACM0')
vehicle = connect('/dev/ttyACM0', wait_ready=True)

vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

while not vehicle.armed:
    print("Ждем моторы...")
    time.sleep(1)

def stop(duration):
    vehicle.channels.overrides['1'] = 1500
    vehicle.channels.overrides['3'] = 1100

def circle(duration):
    vehicle.channels.overrides['1'] = 1300
    vehicle.channels.overrides['3'] = 1100
    
def move_left(duration):
    vehicle.channels.overrides['1'] = 1300
    vehicle.channels.overrides['3'] = 1200
    time.sleep(0.25)

def move_forward(duration):
    vehicle.channels.overrides['3'] = 1400
    vehicle.channels.overrides['1'] = 1500

def fast_move_forward(duration):
    vehicle.channels.overrides['3'] = 2200
    vehicle.channels.overrides['1'] = 1500

def move_right(duration):
    vehicle.channels.overrides['1'] = 1700
    vehicle.channels.overrides['3'] = 1200
    time.sleep(0.25)

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    image = cv2.resize(image, (640, 480))
    height, width = image.shape[:2]
    top_cutoff = int(height * 0.01)
    bottom_cutoff = int(height * 0.66)
    image = image[top_cutoff:bottom_cutoff, :]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bl = cv2.medianBlur(gr, 5)
    canny = cv2.Canny(bl, 10, 250)
    kernal = np.ones((5, 5), "uint8")

    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(image, image,
                               mask=blue_mask)

    contours = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    for cont in contours:
        area = cv2.contourArea(cont)
        sm = cv2.arcLength(cont, True)
        apd = cv2.approxPolyDP(cont, 0.02 * sm, True)

        if len(apd) == 8 and area > 1500:
            cv2.drawContours(image, [apd], -1, (0, 255, 0), 8)
            print("Обнаружен синий восьмиугольник")
            print("Поворот вправо")
            move_right(1)

        elif len(apd) == 5 and area > 1500:
            cv2.drawContours(image, [apd], -1, (0, 255, 0), 5)
            print("Обнаружен синий пятиугольник")
            print("fast")
            fast_move_forward(2)

        elif len(apd) == 4 and area > 1500:
            cv2.drawContours(image, [apd], -1, (0, 255, 0), 4)
            print("Обнаружен синий квадрат")
            print("движение вперед")
            move_forward(1)

        elif len(apd) == 3 and area > 1500:
            cv2.drawContours(image, [apd], -1, (0, 255, 0), 3)
            print("Обнаружен синий треугольник")
            print("Поворот влево")
            move_left(1)
            
        else:
            print("STOP")
            stop(0.25)

    cv2.imshow("Image", image)

    if cv2.waitKey(1) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
