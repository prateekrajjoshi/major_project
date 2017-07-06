import cv2
import os.path
import time
from random import randint

rannum= randint(1,4)
def loadimages():

	print("Waiting for raspberry pi to transmit images")
	while not os.path.exists("crop4.jpg"):
		print("...")
		time.sleep(2)

def makefile(num):
	file= open("signal.txt","w")
	file.write(str(num))
	file.close()
	time.sleep(1)

def signaltransmit():
	print("Transmittig signal to raspberry pi")
	time.sleep(1)
	os.system("scp -r /home/prateek/Desktop/from_raspberry/signal.txt pi@192.168.10.6:/home/pi/Desktop/Prateek")
	print("Transmission Complete")

def deleteeverything():
	os.system("sudo rm crop1.jpg crop2.jpg crop3.jpg crop4.jpg")

loadimages()
makefile(rannum)
print rannum
signaltransmit()
deleteeverything()
