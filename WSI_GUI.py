from tkinter import messagebox
import tkinter
from tkinter import *
import PIL.Image, PIL.ImageTk
from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk
from tkinter import filedialog
from src.App import *





def main():
    print("Hello World!")
    root=ThemedTk(theme="radiance")
    root.geometry('1250x650')
    folder_path = StringVar()
    icon=tkinter.PhotoImage(file="./resource/logo.PNG")
    root.iconphoto(False,icon)
    App(root, "WSI Scanner")

if __name__ == "__main__":
    main()

# Create a window and pass it to the Application object


