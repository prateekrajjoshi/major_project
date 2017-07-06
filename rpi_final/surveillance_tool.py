#!/usr/bin/env python
from operation import image_capture,image_transmit,cam_init,final_works
import sys
import os
import time

try:
	import gi
	gi.require_version('Gtk', '3.0')
	from gi.repository import Gtk
except:
	sys.exit(1)


class loading_help_page:

	wTree = None




	def __init__( self ):
                self.gladefile = "./resources/help_page.glade"
                self.glade = Gtk.Builder()
                self.glade.add_from_file(self.gladefile)
                self.glade.connect_signals(self)
                self.abc=self.glade.get_object("window1")
                self.abc.show_all()
		self.entryForText = self.glade.get_object("textview1")
		with open('./resources/help', 'r') as myfile:
    			data=myfile.read()
		self.entryForText.get_buffer().set_text(data)


	def quit_clicked(self, widget):
		self.abc.hide()




class raspberry_viewer:

	wTree = None

	def __init__( self ):
		self.gladefile = "./resources/rpi_final_design.glade"
                self.glade = Gtk.Builder()
                self.glade.add_from_file(self.gladefile)
                self.glade.connect_signals(self)
                self.window_main = self.glade.get_object("window1")
		self.window_main.resize(640,480)
		self.window_main.show_all()
		self.glade.get_object("label2").set_text("Start a session by clicking to start surveillance")
		self.mainImage = self.glade.get_object("image1")
                self.mainImage.set_from_file("./resources/apple.jpg")
   		self.mainImage.show()


	def quit_clicked(self, widget):
		sys.exit(0)

	def start_clicked(self,widget):
                self.text_setting(self)
                try:
                        number_of_images = 200
                        counter = 0
                        cam_init()
                        while counter < number_of_images:
                                image_capture()
                                image_transmit()
                                counter += 5
                        final_works()
                except KeyboardInterrupt:
                        final_works()   

	def text_setting(self,widget):
		self.glade.get_object("label2").set_text("Image capture and transmission")


	def stop_clicked(self,widget):
		print "abc" #need to add later

	def help_clicked(self,widget):
		open_help = loading_help_page()

	def gtk_main_quit(self, widget):
		sys.exit(0)


launch_gui = raspberry_viewer()
Gtk.main()
