from tkinter import messagebox
import tkinter
from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import time
import numpy as np
from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk
import serial
from tkinter import filedialog
#libraries for stitching 
#Importing the necessory Libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2hsv
from skimage import io
from skimage import data
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import os.path
import glob
import VideoCapture

class App:
    
    def __init__(self, window, window_title, video_source=2):
        self.ser = serial.Serial('/dev/ttyACM0', baudrate = 9600, timeout = 1)
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
     # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.logo1=tkinter.PhotoImage(file="./resource/a.png")
        self.labe2 = tkinter.Label(self.window,image=self.logo1,text='NUST', fg="green", font=('Ariel Bold',30))
        self.labe2.grid(row=0,column=0,columnspan=3,rowspan=2,sticky = tkinter.NW)
        self.logo=tkinter.PhotoImage(file="./resource/b.png")
        

        self.label = tkinter.Label(self.window,image=self.logo,text='Welcome to WSI Program', fg="green", font=('Ariel Bold',30))
        self.label.grid(row=0,column=3,columnspan=5,sticky = tkinter.N)
        self.logo2=tkinter.PhotoImage(file="./resource/c.png")
        self.label = tkinter.Label(self.window,image=self.logo2,text='SIGMA LAB', fg="black", font=('Ariel Bold',25))
        self.label.grid(row=0,column=8,sticky = tkinter.W)

    #  # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.grid(row=1,column=3,rowspan=10,columnspan=5,sticky=tkinter.N)
        self.btn1=ttk.Button(self.window,text="Scan",command=self.scan_slide,width=18)
        self.btn1.grid(row=1,column=0,sticky=tkinter.SE)
        self.btn1=ttk.Button(self.window,text="Focus",command=self.optimal_focus,width=18)
        self.btn1.grid(row=1,column=2,sticky=tkinter.SW)
        var1=tkinter.IntVar()
        self.rad1=ttk.Radiobutton(self.window,text="Microstepping z",command=self.microstepping_z,variable=var1, value=1,width=15 )
        self.rad1.grid(row=2,column=0,sticky=tkinter.S)
        self.rad2=ttk.Radiobutton(self.window,text="Macrostepping z",command=self.macrostepping_z,variable=var1,value=2,width=15 )
        self.rad2.grid(row=3,column=0,sticky=tkinter.N)
        var2=tkinter.IntVar()
        self.rad3=ttk.Radiobutton(self.window,text="Microstepping-xy",command=self.microstepping_xy,variable=var2,value=1,width=15 )
        self.rad3.grid(row=2,column=2,sticky=tkinter.S)
        self.rad4=ttk.Radiobutton(self.window,text="Macrostepping-xy",command=self.macrostepping_xy,variable=var2,value=2,width=15 )
        self.rad4.grid(row=3, column=2,sticky=tkinter.N)
        button1=ttk.Button(self.window, text="leftx",command=self.leftx,width=13)
        button1.grid(row=5,column=0,sticky=tkinter.E)
        button2=ttk.Button(self.window, text="upy",command=self.upy,width=13)
        button2.grid(row=4,column=1,sticky=tkinter.S)
        button3=ttk.Button(self.window, text="rightx",command=self.rightx,width=13)
        button3.grid(row=5,column=2,sticky=tkinter.W)
        button4=ttk.Button(self.window, text="downy",command=self.downy,width=13)
        button4.grid(row=6,column=1,sticky=tkinter.N)
        button10 =ttk.Button(self.window, text='up',command=self.upz,width=8)
        button10.grid(row=4,column=2,sticky=tkinter.E)
        button12 = ttk.Button(self.window, text='down',command=self.downz,width=8)
        button12.grid(row=6, column=2,sticky=tkinter.E)
        self.text_box = tkinter.Text(self.window,spacing2=5,relief=tkinter.GROOVE,insertborderwidth=5,width=27,height=30,highlightbackground="black", highlightcolor="gray56", highlightthickness=3)
        self.text_box.grid(row=1,column=8,rowspan=10,sticky=tkinter.NW)
        button14=ttk.Button(self.window, text="Stitching",command=self.lets_stitch_images,width=16)
        button14.grid(row=7,column=0,sticky=tkinter.SE )
        button15=ttk.Button(self.window, text="Detection",command=self.my_detection_code,width=14)
        button15.grid(row=7,column=1,sticky=tkinter.S  )
        button16=ttk.Button(self.window, text="Origin",command=self.origin,width=16)
        button16.grid(row=7,column=2,sticky=tkinter.SW )
        button13=ttk.Button(self.window, text="Exit",command=self.ExitApplication,width=49)
        button13.grid(row=8,column=0,columnspan=3,sticky=tkinter.N )
        self.label4 = tkinter.Label(self.window,text='Low Cost Whole Slide Image Scanner\n Advisor: Engr Imran Abeel\nCo-Advisor: Dr Faisal Shafiat', fg="black", font=('Ariel Bold',15))
        self.label4.grid(row=9,column=0,columnspan=3,sticky = tkinter.SW)


        self.window.protocol("WM_DELETE_WINDOW", self.ExitApplication)
        self.update()
        self.window.mainloop()

    def update(self):
         # Get a frame from the video source
         while 1:
             ret, frame = self.vid.get_frame()
             if ret:
                 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
             else:
                 return
             self.window.update()

    def origin(self):
        self.text_box.insert(tkinter.INSERT,"moving to origin\n")
        self.text_box.see("end")
        ser.write(b'o')

    def ExitApplication(self):
        MsgBox =messagebox.askquestion("Exit Application","Are you sure you want to exit the application")
        if MsgBox == 'yes':
            self.quit()
        else:
            messagebox.showinfo("Return","You will now return to the application screen")

    def horizontal_strip_with_framecapture(self, direction):
        if (direction=='left_4'):
            self.ser.write(b'8')
            self.text_box.insert(tkinter.INSERT,"left\n")
            self.text_box.see("end")
        else:
            self.ser.write(b'2')
            self.text_box.insert(tkinter.INSERT,"right\n")
            self.text_box.see("end")

    def scan_slide(self):
        self.text_box.insert(tkinter.INSERT,"scanning\n")
        self.text_box.see("end")
        sec = 0
        frameRate = 0.5
        total_sets=4  #predefined in the rectangular slide
        self.vid.__del__()
        cap = cv2.VideoCapture(2)
        x=0

        for i in range(total_sets):
            self.optimal_focus()                   
            #Now use left x
            direction='left_4'
            self.horizontal_strip_with_framecapture(direction)
            #now start capturing frames as well
            

            while(self.ser.readline()[:-2] != b'scanning_ended'):
                # sec = sec + frameRate
                # sec = round(sec, 2)
                _, frame = cap.read()
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = photo, anchor = tkinter.NW)
                self.window.update()
                sec=sec+1
                if not sec%1:
                        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        cv2.imwrite(r'./stitching_frames/pic'+str(sec)+r'.jpg',frame)
                if sec==40:
                    break
            
##                
            self.optimal_focus()
            self.upy() #vertical always in down direction

            while(self.ser.readline()[:-2] != b'scanning_ended'):
                # sec = sec + frameRate
                # sec = round(sec, 2)
                _, frame = cap.read()
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = photo, anchor = tkinter.NW)
                self.window.update()
                sec=sec+1
                if not sec%1:
                        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        cv2.imwrite(r'./stitching_frames/pic'+str(sec)+r'.jpg',frame)
##            
            self.optimal_focus()
             #Now use right  x
            direction='right_6'
            self.horizontal_strip_with_framecapture(direction)  
            while(self.ser.readline()[:-2] != b'scanning_ended'):
                _, frame = cap.read()
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = photo, anchor = tkinter.NW)
                self.window.update()
                sec=sec+1
                if not sec%1:
                        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        cv2.imwrite(r'./stitching_frames/pic'+str(sec)+r'.jpg',frame)


            self.upy()  #vertical always in down direction
            while(self.ser.readline()[:-2] != b'scanning_ended'):
                _, frame = cap.read()
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image = photo, anchor = tkinter.NW)
                self.window.update()
                sec=sec+1
                if not sec%1:
                        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        cv2.imwrite(r'./stitching_frames/pic'+str(sec)+r'.jpg',frame)
    
        cap.release()
        self.video_obj(self.video_source)
        self.text_box.insert(tkinter.INSERT,"scanning ended\n")
        self.text_box.see("end")
        return
                   


    def optimal_focus(self):
            self.vid.__del__()
            global_option=True
            self.text_box.insert(tkinter.END,"Focusing\n")
            self.text_box.see("end")
            x=0
            cap = cv2.VideoCapture(2)
            edges_current = 0
            edges_previous = 0
            frame_count = 0
            count=0
            direction = True
            uudd_array=[]
            dduu_array=[]
            ududud_array=[]
            dududu_array=[]
            # self.ser.write(b'0')
            damp_count=0
            while cap.isOpened():
                    _, frame = cap.read()
                    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                    self.canvas.create_image(0, 0, image = photo, anchor = tkinter.NW)
                    self.window.update()
                    HSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                    edges=np.sum(cv2.Laplacian(HSV[:,:,1],cv2.CV_64F).var())

                    
                    if frame_count==20:
                        edges_current = np.sum(edges)
                        edges_previous = (edges_previous)
                        edges_current = (edges_current)
                        print(edges_previous, "  ", edges_current)
                        if (edges_current>edges_previous)and direction:
                            
                            uudd_array.append(1)
                            self.ser.write(b'7')
                            print(edges_current-edges_previous)
                            print('move up')
                        elif (edges_current<edges_previous)and direction:
                            uudd_array.append(0)
                            self.ser.write(b'9')
                            direction = False
                            print(edges_current-edges_previous)
                            print('move down')
                        elif (edges_current>edges_previous)and not(direction):
                            uudd_array.append(0)
                            self.ser.write(b'9')
                            print(edges_current-edges_previous)
                            print('move down')
                        elif (edges_current<edges_previous)and not(direction):
                            uudd_array.append(1)
                            self.ser.write(b'7')
                            direction  = True
                            print(edges_current-edges_previous)
                            print('move up')
                        frame_count = 0
                        edges_previous = edges_current
            #or uudd_array==[1,1,1,0] or uudd_array==[1,1,0,0] or uudd_array==[1,0,0,0] or uudd_array==[0,0,0,1] or uudd_array==[0,0,1,1] or uudd_array==[0,1,1,1]
                    #check detected sequence
                    if(global_option==True):
                        if (len(uudd_array)==4):
                            print('The Sequence check array is given as ' +str(uudd_array))
                            if (uudd_array==[0,0,1,1] or uudd_array==[0,1,1,0] or uudd_array==[1,1,0,0] or uudd_array==[1,0,0,1] or uudd_array==[0,0,0,1] or uudd_array==[1,1,1,0]): 
                                damp_count=damp_count+1
                                print( 'came here sequce match ' +str( damp_count))
                                if damp_count==5:
                                    print('DAMMPNG HAS STARTED')
                                    self.ser.write(b'1')
                                    global_option=False
                                    
                                    damp_count=0;
                            else:
                                    damp_count=0
                            del uudd_array[0]
                            del uudd_array[0]
                            del uudd_array[0]
                            del uudd_array[0]
                            cv2.imwrite('my_img.jpg',frame)
                    else:
                        if (len(uudd_array)==4):
                            print('The Sequence check array is given as ' +str(uudd_array))
                            if (uudd_array==[0,0,1,1] or uudd_array==[0,1,1,0] or uudd_array==[1,1,0,0] or uudd_array==[1,0,0,1] or uudd_array==[1,1,1,0] or uudd_array==[1,1,0,0] or uudd_array==[1,0,0,0] or uudd_array==[0,0,0,1] or uudd_array==[0,0,1,1] or uudd_array==[0,1,1,1] or uudd_array==[0,0,1,0] or uudd_array==[1,0,1,1]): 
                                damp_count=damp_count+1
                                print( 'came here sequce match ' +str( damp_count))
                                if damp_count==5:
                                    print('DAMMPNG HAS STARTED')
                                    # self.ser.write(b'1')
                                    cv2.imwrite('TEST_IMG.tiff',frame)
                                    cap.release()
                                    self.video_obj(self.video_source)
                                    return
                                    
                                    damp_count=0;
                            else:
                                    damp_count=0
                            del uudd_array[0]
                            del uudd_array[0]
                            del uudd_array[0]
                            del uudd_array[0]
                            cv2.imwrite('my_img.jpg',frame)


                        
                    
                    count=count+1
                    frame_count=frame_count+1
                    k=cv2.waitKey(5) & 0xFF
                    
                    if k==27:
                        break
            cv2.destroyAllWindows()
            cap.release()
            self.video_obj(self.video_source)
            return
                

     


    def microstepping_z(self):
        self.text_box.insert(tkinter.END,"Microstepping_Z\n")
        self.text_box.see("end")
        self.ser.write(b'1')
        
    
    def macrostepping_z(self):
        self.text_box.insert(tkinter.END,"Macrostepping_Z\n")
        self.text_box.see("end")
        self.ser.write(b'0')
        
    def upy(self):
        self.text_box.insert(tkinter.END,"Moving Up in Y axis\n")
        self.text_box.see("end")
        self.ser.write(b'6')
        
    def macrostepping_xy(self):
        self.text_box.insert(tkinter.END,"Macrostepping_xy\n")
        self.text_box.see("end")
        self.ser.write(b'3')
        

    def rightx(self):
        self.text_box.insert(tkinter.END,"Moving Right in X axis\n")
        self.text_box.see("end")
        self.ser.write(b'2')
        
    def microstepping_xy(self):
        self.text_box.insert(tkinter.END,"Microstepping_xy\n")
        self.text_box.see("end")
        self.ser.write(b'5')
        
    def leftx(self):
        self.text_box.insert(tkinter.END,"Moving Left in X axis\n")
        self.text_box.see("end")
        self.ser.write(b'8')
        
    def upz(self):
        self.text_box.insert(tkinter.END,"Up\n")
        self.text_box.see("end")
        self.ser.write(b'9')
    
    def downy(self):
        self.text_box.insert(tkinter.END,"Moving Down in Y axis \n")
        self.text_box.see("end")
        self.ser.write(b'4')
        
    def downz(self):
        self.text_box.insert(tkinter.END,"Down\n")
        self.text_box.see("end")
        self.ser.write(b'7')



    def quit(self):
        self.vid.__del__()
        root.deiconify()
        self.canvas.destroy()
        root.destroy()


    def update(self):
         # Get a frame from the video source
         while 1:
             ret, frame = self.vid.get_frame()
             if ret:
                 self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
                 self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
             else:
                 return
             self.window.update()
         
    def g_frame(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()
         self.text_box.insert(tkinter.END,"frm\n")
         self.text_box.see("end")
         return frame
        
    def video_obj(self,video_source=0):
         self.vid = MyVideoCapture(self.video_source)
         return self.vid

    #function to implement the laplacian blending between overlapping regions during stitching
    def gaussian_pyramid(self,img, num_levels):
        lower = img.copy()
        gaussian_pyr = [lower]
        for i in range(num_levels):
            lower = cv2.pyrDown(lower)
            gaussian_pyr.append(np.float32(lower))
        return gaussian_pyr

    #function to implement the laplacian blending between overlapping regions during stitching
    def laplacian_pyramid(self,gaussian_pyr):
        laplacian_top = gaussian_pyr[-1]
        num_levels = len(gaussian_pyr) - 1
        
        laplacian_pyr = [laplacian_top]
        for i in range(num_levels,0,-1):
            size = (gaussian_pyr[i - 1].shape[1], gaussian_pyr[i - 1].shape[0])
            gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
            laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
            laplacian_pyr.append(laplacian)
        return laplacian_pyr
    
    #function to implement the laplacian blending between overlapping regions during stitching
    def blend(self,laplacian_A,laplacian_B,mask_pyr):
        LS = []
        for la,lb,mask in zip(laplacian_A,laplacian_B,mask_pyr):
            ls = lb * mask + la * (1.0 - mask)
            LS.append(ls)
        return LS


    #function to implement the laplacian blending between overlapping regions during stitching
    def reconstruct(self,laplacian_pyr):
        laplacian_top = laplacian_pyr[0]
        laplacian_lst = [laplacian_top]
        num_levels = len(laplacian_pyr) - 1
        for i in range(num_levels):
            size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
            laplacian_expanded = cv2.pyrUp(laplacian_top, dstsize=size)
            laplacian_top = cv2.add(laplacian_pyr[i+1], laplacian_expanded)
            laplacian_lst.append(laplacian_top)
        return laplacian_lst

    def check_arguments_errors(self,args):
        assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(args.config_file):
            raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
        if not os.path.exists(args.weights):
            raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
        if not os.path.exists(args.data_file):
            raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
        if args.input and not os.path.exists(args.input):
            raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))


    def check_batch_shape(self,images, batch_size):
        """
            Image sizes should be the same width and height
        """
        shapes = [image.shape for image in images]
        if len(set(shapes)) > 1:
            raise ValueError("Images don't have same shape")
        if len(shapes) > batch_size:
            raise ValueError("Batch size higher than number of images")
        return shapes[0]


    def load_images(self,images_path):
        """
        If image path is given, return it directly
        For txt file, read it and return each line as image path
        In other case, it's a folder, return a list with names of each
        jpg, jpeg and png file
        """
        input_path_extension = images_path.split('.')[-1]
        if input_path_extension in ['jpg', 'jpeg', 'png']:
            return [images_path]
        elif input_path_extension == "txt":
            with open(images_path, "r") as f:
                return f.read().splitlines()
        else:
            return glob.glob(
                os.path.join(images_path, "*.jpg")) + \
                glob.glob(os.path.join(images_path, "*.png")) + \
                glob.glob(os.path.join(images_path, "*.jpeg"))


    def prepare_batch(self,images, network, channels=3):
        width = darknet.network_width(network)
        height = darknet.network_height(network)

        darknet_images = []
        for image in images:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
            custom_image = image_resized.transpose(2, 0, 1)
            darknet_images.append(custom_image)

        batch_array = np.concatenate(darknet_images, axis=0)
        batch_array = np.ascontiguousarray(batch_array.flat, dtype=np.float32)/255.0
        darknet_images = batch_array.ctypes.data_as(darknet.POINTER(darknet.c_float))
        return darknet.IMAGE(width, height, channels, darknet_images)


    def image_detection(self,image_path, network, class_names, class_colors, thresh):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        darknet_image = darknet.make_image(width, height, 3)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


    def batch_detection(self,network, images, class_names, class_colors,
                        thresh=0.25, hier_thresh=.5, nms=.45, batch_size=4):
        image_height, image_width, _ = check_batch_shape(images, batch_size)
        darknet_images = self.prepare_batch(images, network)
        batch_detections = darknet.network_predict_batch(network, darknet_images, batch_size, image_width,
                                                        image_height, thresh, hier_thresh, None, 0, 0)
        batch_predictions = []
        for idx in range(batch_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if nms:
                darknet.do_nms_obj(detections, num, len(class_names), nms)
            predictions = darknet.remove_negatives(detections, class_names, num)
            images[idx] = darknet.draw_boxes(predictions, images[idx], class_colors)
            batch_predictions.append(predictions)
        darknet.free_batch_detections(batch_detections, batch_size)
        return images, batch_predictions


    def image_classification(self,image, network, class_names):
        width = darknet.network_width(network)
        height = darknet.network_height(network)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.predict_image(network, darknet_image)
        predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
        darknet.free_image(darknet_image)
        return sorted(predictions, key=lambda x: -x[1])


    def convert2relative(self,image, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        height, width, _ = image.shape
        return x/width, y/height, w/width, h/height


    def save_annotations(self,name, image, detections, class_names):
        """
        Files saved with image_name.txt and relative coordinates
        """
        file_name = name.split(".")[:-1][0] + ".txt"
        with open(file_name, "w") as f:
            for label, confidence, bbox in detections:
                x, y, w, h = self.convert2relative(image, bbox)
                label = class_names.index(label)
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))


    def batch_detection_example(self):
        args = parser()
        self.check_arguments_errors(args)
        batch_size = 3
        random.seed(3)  # deterministic bbox colors
        network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=batch_size
        )
        image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
        images = [cv2.imread(image) for image in image_names]
        images, detections,  = self.batch_detection(network, images, class_names,
                                            class_colors, batch_size=batch_size)
        for name, image in zip(image_names, images):
            cv2.imwrite(name.replace("data/", ""), image)
        print(detections)


    def my_detection_code(self):
        print("Yolo detection code has been removed. Please integrate your code here")


    def lets_stitch_images(self):
        global folder_path
        filename = filedialog.askdirectory(title="Please choose folder where images are stored")
        folder_path.set(filename)
        if filename== () or filename== "":
            return
        my_list=glob.glob(filename+ '//*')
        my_list
        #list_names contains drive path of each image, which are sequentially stored.
        
        list_names=sorted(my_list)       
        print(list_names)
        empty=np.zeros((2800,5800,3))
        empty=empty.astype(np.uint8)

        #No of levels of pyramaid of the Image picture
        num_levels = 7

        last_height_position=500
        last_width_position=5000
        #creating variables for cropping purpose at the last step after stitching
        #remember height rules will be inverse
        left_most_minimum_width_position=last_width_position
        right_most_maximum_width_position=last_width_position
        maximum_height_position=last_height_position
        minimum_height_position=last_height_position
        #first time we need to place a full image at start
        im=io.imread(list_names[0])
        empty[last_height_position:last_height_position+im.shape[0],last_width_position:im.shape[1]+last_width_position,:]=im
        last_width_position=im.shape[1]+last_width_position
        #This is the main loop where all the upcoming images are stitched together
        #These indexes used to ensure that any two images which needs to be stitched has overlap as less as possible
        idx1 = int(0)
        idx2 = int(1)
        x=0
        for i in range(len(list_names)-3):
            #read the first two images (consecutive images)
            print(idx1,idx2)
            im1=io.imread(list_names[idx1])
            im2=io.imread(list_names[idx2])
            
            #using cross co-relation find the offset between the images in both X and Y direction
            shift, error, diffphase = register_translation(im1, im2)
            if(abs(shift[0]) < 100 and abs(shift[1]) < 100):
                idx2+=1
            else:
                print("offset x, ys = ", shift[0], ", ", shift[1])
                #update the position pointers with the generated offset
                last_height_position=last_height_position+int(shift[0])
                last_width_position=int(shift[1])+last_width_position
                #make the probablity mask image where in which the area will be 1 where
                #there is overlap of image with the existing area of empty array (where image 2 has to be pasted fully)
                p = np.where(empty[last_height_position:last_height_position+im2.shape[0],last_width_position-im2.shape[1]:last_width_position,:] > 1, 0, 1)
                p = p.astype(np.float32)
                #only paste the Non-overlapping area in the big empty array
                empty[last_height_position:last_height_position+im2.shape[0],last_width_position-im2.shape[1]:last_width_position,:] += (p*im2).astype(np.uint8)
                #now extract the region where Im2 needs to be fully pasted and expand that image 100 pixels in each direction
                a = copy.copy(empty[last_height_position-100:last_height_position+im2.shape[0]+100,last_width_position-im2.shape[1]-100:last_width_position+100,:])
                #now completely paste image2 in its location in the empty array
                empty[last_height_position:last_height_position+im2.shape[0],last_width_position-im2.shape[1]:last_width_position,:] = im2
                #now again extract the region where Im2 needs to be fully pasted (which is now pasted) and expand that image 100 pixels in each direction
                b = copy.copy(empty[last_height_position-100:last_height_position+im2.shape[0]+100,last_width_position-im2.shape[1]-100:last_width_position+100,:])
                #now we have the source and target image, mask generation is needed only now
                m = np.zeros(a.shape).astype(np.float32)
                #make it white where overlapping area exists
                m[100:-150,150:-150,:] = 1
                #now perform the laplacian blendind technieque
                gaussian_pyr_1 = self.gaussian_pyramid(a, num_levels)
                laplacian_pyr_1 = self.laplacian_pyramid(gaussian_pyr_1)
                gaussian_pyr_2 = self.gaussian_pyramid(b, num_levels)
                laplacian_pyr_2 = self.laplacian_pyramid(gaussian_pyr_2)
                mask_pyr_final = self.gaussian_pyramid(m, num_levels)
                mask_pyr_final.reverse()
                add_laplace = self.blend(laplacian_pyr_1,laplacian_pyr_2,mask_pyr_final)
                final  = self.reconstruct(add_laplace)
                reconstructed = final[num_levels]
                reconstructed = reconstructed.astype(np.uint8)
                #now paste the complete blended result into main empty array.
                empty[last_height_position-100:last_height_position+im2.shape[0]+100,last_width_position-im2.shape[1]-100:last_width_position+100,:] = reconstructed 
                idx1 = idx2
                idx2 = idx1 + 1
            #now updating variables fo cropping
            if (shift[1]<0):
                if (left_most_minimum_width_position>last_width_position-im2.shape[1]):
                    left_most_minimum_width_position=last_width_position-im2.shape[1]
            if (right_most_maximum_width_position<last_width_position):
                right_most_maximum_width_position=last_width_position
            if (maximum_height_position)<last_height_position+im2.shape[0]:
                maximum_height_position=last_height_position+im2.shape[0]
            if (minimum_height_position>last_height_position):
                minimum_height_position=last_height_position
            x=x+1  
            print(last_width_position)
            if(x==3):
                x=0

                plt.ion()
                plt.show()
                plt.imshow(empty)
                plt.pause(0.001)
                
