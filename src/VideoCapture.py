import cv2
import PIL.Image, PIL.ImageTk


class MyVideoCapture:
     def __init__(self, video_source=2):
         # Open the video source
         self.vid = cv2.VideoCapture(video_source)
##         print("vid")
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)

         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (False, None)

     def write_frames(self,sec):
         self.vid.set(cv2.CAP_PROP_POS_MSEC,sec*0)
         if self.vid.isOpened():
              ret, frame = self.vid.read()
              if ret:
                  cv2.imwrite(r'./stitching_frames/pic'+str(sec)+r'.jpg',frame)
         return ret


     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()

