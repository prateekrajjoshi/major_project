#!/usr/bin/env python
import sys
import os
import time
import picamera

FRAMES = 5 #for no. of images change value here
TIMEBETWEEN = 6 #for delay increase value here

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
            camera.capture('./images_captured/image%s.jpg'%(imageNumber))
        except:
            print "Error in image capture!"
            sys.exit(1)

        frameCount += 1
        time.sleep(TIMEBETWEEN - 6) #Takes roughly 6 seconds to take a picture
        

def image_transmit():
    #transferring captured images from raspberry pi to server(my pc in this case)
    number_of_images = FRAMES 
    num_count =0
    WAIT_FOR_SCP_TRANSMIT = 6
    while num_count < number_of_images:
            imageNumber = str(num_count).zfill(7)
            try:
                    os.system("scp -r /home/pi/Desktop/new_one/images_captured/image%s.jpg pranjal@192.168.0.110:/home/pranjal/Desktop/images_pi"%(imageNumber))
                    #time.sleep(WAIT_FOR_SCP_TRANSMIT - 3) #Taking roughly 3 seconds to transmit image

            except Exception as abc:
                    print abc
                    sys.exit(1)
            num_count+=1

def final_works():
    frameCount = 0
