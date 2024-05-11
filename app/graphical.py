import tkinter as tk
from tkinter import filedialog, ttk  # Import ttk for combobox
from tkinter import filedialog
from segmentation_client import upload_for_segmentation
import numpy as np
# Function to handle file selection
def select_file():
  global file_path
  file_path = filedialog.askopenfilename(title="Select Medical Data File", filetypes=[("Nii Files", "*.nii")])
  if file_path:
    file_var.set(file_path)

# Function to initiate segmentation
def start_segmentation():
  global file_path
  mask = upload_for_segmentation(file_path, server_url=server_url_var.get())
  if mask is not None:
    # Add logic to process or display the segmentation mask (e.g., new window)
    print("Segmentation completed!")
    np.save('app/Predictions/mask.npy', mask)
  else:
    print("Error during upload or processing.")

# Initialize the main window
root = tk.Tk()
root.title("Segmentation Client")
root.geometry("600x400")  # Set width and height in pixels

# Server URL selection with dropdown menu and custom entry
server_url_options = ["http://localhost:8000/seg3dtest/", "http://your-server1.com:8080/segment", "http://your-server2.com:5000/segmentation"]
server_url_var = tk.StringVar()
server_url_var.set(server_url_options[0])  # Set default selection

server_url_label = tk.Label(root, text="Server URL:")
server_url_label.pack()
server_url_menu = ttk.Combobox(root, values=server_url_options + ["custom"], textvariable=server_url_var)
server_url_menu.pack()

# Custom URL entry field (appears when "custom" is selected)
custom_url_var = tk.StringVar(root)
custom_url_label = tk.Label(root, text="Custom URL:")
custom_url_label.pack(padx=5)  # Add some padding for better layout
custom_url_entry = tk.Entry(root, textvariable=custom_url_var, state='disabled')  # Initially disabled
custom_url_entry.pack()

def handle_url_selection(event):
  # Enable/disable custom URL entry based on selection
  if server_url_var.get() == "custom":
    custom_url_entry.config(state='normal')
  else:
    custom_url_entry.config(state='disabled')

server_url_menu.bind("<<ComboboxSelected>>", handle_url_selection)  # Bind event handler


# File path display and selection button
file_var = tk.StringVar()
file_label = tk.Label(root, text="File Path:")
file_label.pack()
file_path_entry = tk.Entry(root, textvariable=file_var, state='readonly')
file_path_entry.pack()
select_file_button = tk.Button(root, text="Select File", command=select_file)
select_file_button.pack()

# Start segmentation button
start_button = tk.Button(root, text="Start Segmentation", command=start_segmentation)
start_button.pack()

# Run the main loop
root.mainloop()
