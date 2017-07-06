'''
Sample code using numpy to convert image to array
'''

import numpy
from PIL import Image
from numpy import array
import os
import random

def main(path):
	count = 0
	image_list = []
	dir_list = os.listdir(path)
	dir_list.sort()
	print len(dir_list)
	for f in dir_list :
		#print f
		os.chdir(path)
		os.chdir('../')
		print os.getcwd()
		print os.path.split(path)
		if not os.path.exists(os.path.join(os.getcwd(), (os.path.split(path)[-1] + '_new'))):
			os.makedirs(os.path.join(os.getcwd(), (os.path.split(path)[-1] + '_new')))
		path_n = os.path.join(os.getcwd(), os.path.split(path)[-1] + '_new')	
		os.chdir(path_n)
		#if not os.path.exists(os.path.join(os.getcwd(), os.path.split(f)[-1])):
		if not os.path.exists(os.path.join(os.getcwd(),f)):
			os.makedirs(os.path.join(os.getcwd(), f))
		path_i = os.path.join(os.getcwd(), f)
		#print path_i
		os.chdir(os.path.join(path,f))

		for v in os.listdir(os.path.join(path, f)):
			img = Image.open(v)
			width, height = img.size
			#print img
			
			# crop out the center 300x300
			#crop_h = (width-150)/2
			#crop_v = (height-150)/2
			
			# COMMENT THIS TO AVOID CROPPING
			#img = img.crop((crop_h, crop_v, width-crop_h, height-crop_v))
			# resize the ex image to 150x150
			#img = img.resize((150, 150))

			# convert to grayscale
			img = img.convert('L')
		
			img.save(os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count)))
			image_list.append((os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count))) + ' ' + str(dir_list.index(f)))	
			count += 1

			img2 = img.rotate(180 , expand=True)
			img2.save(os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count)))
			image_list.append((os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count))) + ' ' + str(dir_list.index(f)))	
			count += 1
			'''
			img2 = img.rotate(45 , expand=True)
			img2.save(os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count)))
			image_list.append((os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count))) + ' ' + str(dir_list.index(f)))	
			count += 1

			img2 = img.rotate(-45 , expand=True)
			img2.save(os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count)))
			image_list.append((os.path.join(path_i, os.path.split(path_i)[-1] + '_img%d.jpg' % (count))) + ' ' + str(dir_list.index(f)))	
			count += 1'''
			#arr = array(img).reshape(150*150)
			#print(arr)
			#print(img)
	#print image_list
	random.shuffle(image_list)
	train_data = image_list[:int((len(image_list)+1)*.80)] #Remaining 80% to training set
	test_data = image_list[int(len(image_list)*.80+1):] #Splits 20% data to test set
	os.chdir(os.path.split(path)[0])
	if os.path.exists('train.txt'):
		os.remove('train.txt')
	if os.path.exists('text.txt'):
		os.remove('test.txt')
	if os.path.exists('num_class.txt'):
		os.remove('num_class.txt')		
	with open('train.txt','w') as f_train:
		for f in train_data:
			f_train.write(f+"\n")
		f_train.close()
	with open('valid.txt','w') as f_test:
		for f in test_data:
			f_test.write( f + "\n")
		f_test.close()
	with open('num_class.txt','w') as num_class:
		num_class.write(str(len(dir_list)))
		num_class.close()


if __name__ == '__main__':
	path = os.path.join(os.getcwd(),'dataSet')
	main(path)
