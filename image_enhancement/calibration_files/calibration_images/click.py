#!/usr/bin/env python
import sys
import os
import time
import picamera

FRAMES = 20
TIMEBETWEEN = 9

def cam_init():    
    #image capture
    global camera
    camera = picamera.PiCamera()
    camera.resolution = (640,480)
    camera.vflip = True

def image_capture():
    frameCount= 0
    while frameCount < FRAMES:
        imageNumber = str(frameCount).zfill(7)
        try:
            camera.capture('imagelack1%s.jpg'%(imageNumber))
        except Exception as exc:
            print "Error in image capture!",exc
            sys.exit(1)

        frameCount += 1
        time.sleep(TIMEBETWEEN - 6) #Takes roughly 6 seconds to take a picture

cam_init()
image_capture()
