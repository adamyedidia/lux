import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

print("Please click on the wall corner.")
# stage 0 is waiting for a click on the wall corner
# stage 1 is waiting for a click on the door corner
# stage 2 is waiting for a click on the part of the ceiling vertically above the door
stage = 0

wallCornerLoc = None
doorCornerLoc = None
vertLoc = None

def on_click(event=None):
    global stage
    global wallCornerLoc
    global doorCornerLoc
    global vertLoc
    # `command=` calls function without argument
    # `bind` calls function with one argument
    if stage == 0:
        print("Click occurred on", event.x, event.y)
        print("Please click on the door corner.")
        stage = 1
        wallCornerLoc = np.array([event.x, event.y])
    elif stage == 1:
        print("Click occurred on", event.x, event.y)
        print("Please click on the part of the ceiling vertically above the door.")
        stage = 2
        doorCornerLoc = np.array([event.x, event.y])
    elif stage == 2:
        print("Click occurred on", event.x, event.y)
        print("You're done. Thanks!")
        vertLoc = np.array([event.x, event.y])
        print("Horizontal vector:", vertLoc - wallCornerLoc)
        print("Vertical vector:", doorCornerLoc - vertLoc)

# --- main ---

# init
root = tk.Tk()

# load image
image = Image.open("doorway_camera.png")
photo = ImageTk.PhotoImage(image)

# label with image
l = tk.Label(root, image=photo)
l.pack()

# bind click event to image
l.bind('<Button-1>', on_click)

# button with image binded to the same function
b = tk.Button(root, image=photo, command=on_click)
b.pack()

# button with text closing window
b = tk.Button(root, text="Close", command=root.destroy)
b.pack()

# "start the engine"
root.mainloop()
