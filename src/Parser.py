import os.path
import glob
import argparse



class parser():
    def __init__(self):
        global folder_path
        self.input = filedialog.askopenfilename(title="Please select image for cells detection",filetypes=[("Image File","*.jpg"),("Image File","*.png")])
        folder_path.set(input)