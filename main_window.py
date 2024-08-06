import tkinter as tk
from tkinter import messagebox
import os
import sys
import threading
import traceback
import subprocess  # For launching the point cloud visualization script
import open3d as o3d  # Ensure open3d is installed and imported if used

sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

# Import the necessary functions (assuming these functions are properly defined in your source)
from Source.fileHandler import get_file_path, readout_LAS_file
from Source.pointCloudAltering import grid_subsampling, remove_noise_statistical
from Interfaces.ThreeDeeprinting_window import App as PrintingApp


class PointCloudApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Processing")
        self.root.geometry("600x350")  # Adjusted height for new buttons
        self.root.resizable(False, False)
        self.root.iconbitmap("Source\\support_files\\logo.ico")

        # Define button size
        self.button_width = 20
        self.button_height = 2

        self.create_widgets()
        self.file_path = None  # Initialize file path variable
        self.point_cloud_data = None  # To store the processed point cloud data
        self.pcd_after_dwnsmpl = None  # To store the downsampled point cloud data
        self.original_downsample_text = "Downsample Pointcloud"  # Store the original button text

        # If the escape key is pressed, activate the on_close method
        self.root.bind("<Escape>", lambda e: self.on_close())

        # Bind the window close event to the on_close method
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Row 0
        self.btn_choose_file = tk.Button(
            self.root, text="Choose a\nLAS/LAZ file", width=self.button_width, height=self.button_height,
            command=self.choose_file
        )
        self.btn_choose_file.grid(row=0, column=0, padx=10, pady=10, sticky='ew')

        self.label_info = tk.Label(self.root, text="No file selected", anchor='w', wraplength=500, justify='left')
        self.label_info.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        # Row 1
        self.entry_field = tk.Entry(self.root, width=self.button_width, state='disabled')
        self.entry_field.grid(row=1, column=0, padx=10, pady=10, sticky='ew')

        self.btn_downsample = tk.Button(
            self.root, text="Downsample Pointcloud", width=self.button_width, height=self.button_height,
            state='disabled', command=self.start_downsample_thread
        )
        self.btn_downsample.grid(row=1, column=1, padx=10, pady=10, sticky='ew')

        # Row 2
        self.btn_remove_noise = tk.Button(
            self.root, text="Remove noise\nfrom pointcloud", width=self.button_width * 2, height=self.button_height,
            state='disabled', command=self.start_remove_noise_thread
        )
        self.btn_remove_noise.grid(row=2, column=0, padx=10, pady=10, sticky='ew')

        self.label_remove_noise_info = tk.Label(self.root, text="", anchor='w', justify='left')
        self.label_remove_noise_info.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Checkbox for visualization
        self.visualize_var = tk.BooleanVar()  # Variable to hold the state of the checkbox
        self.check_visualize = tk.Checkbutton(
            self.root, text="Visualize", variable=self.visualize_var, onvalue=True, offvalue=False
        )
        self.check_visualize.grid(row=2, column=1, padx=10, pady=10, sticky='e')

        # Row 3
        self.btn_repair_save = tk.Button(
            self.root, text="Repair Point cloud\nand save as a new file", width=self.button_width * 2, height=self.button_height,
            state='disabled'
        )
        self.btn_repair_save.grid(row=3, column=0, padx=10, pady=10, sticky='ew')

        self.btn_open_3d_printing = tk.Button(
            self.root, text="Open 3d printing\nmodule", width=self.button_width * 2, height=self.button_height,
            state='disabled', command=self.open_3d_printing_module
        )
        self.btn_open_3d_printing.grid(row=3, column=1, padx=10, pady=10, sticky='ew')

        # Row 4
        self.btn_reset = tk.Button(
            self.root, text="Reset", width=self.button_width, height=self.button_height,
            command=self.reset
        )
        self.btn_reset.grid(row=4, column=0, padx=10, pady=10, sticky='ew')

        self.btn_view_point_cloud = tk.Button(
            self.root, text="View Point Cloud", width=self.button_width * 2, height=self.button_height,
            state='disabled', command=self.start_view_point_cloud_thread
        )
        self.btn_view_point_cloud.grid(row=4, column=1, padx=10, pady=10, sticky='ew')

        # Configure column weights to make buttons expand
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=0)

    def choose_file(self):
        # Update the button text and disable it while processing
        self.update_button_text(self.btn_choose_file, "Loading...")
        self.btn_choose_file.config(state='disabled')

        # Open a file dialog to choose a file
        file_path = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
        if file_path:
            # Read the file
            pcd = readout_LAS_file(file_path)
            if pcd is not None:
                # Store the file path and initial point cloud data
                self.file_path = file_path
                self.point_cloud_data = pcd
                self.update_point_cloud_info(pcd)
                self.enable_buttons_after_file_selection()
            else:
                messagebox.showerror("Error", "Failed to read the LAS/LAZ file.")
        else:
            self.file_path = None
            self.label_info.config(text="No file selected")
            self.disable_buttons()

        # Re-enable the button and restore its original text
        self.update_button_text(self.btn_choose_file, "Choose a\nLAS/LAZ file")
        self.btn_choose_file.config(state='normal')

    def enable_buttons_after_file_selection(self):
        # Enable downsample input field and button
        self.entry_field.config(state='normal')
        self.btn_downsample.config(state='normal')

    def disable_buttons(self):
        # Disable all buttons that should be inactive when no file is selected
        self.btn_downsample.config(state='disabled')
        self.btn_remove_noise.config(state='disabled')
        self.btn_repair_save.config(state='disabled')
        self.btn_open_3d_printing.config(state='disabled')
        self.btn_view_point_cloud.config(state='disabled')
        self.entry_field.config(state='disabled')

    def update_point_cloud_info(self, pcd):
        # Update point count and color information
        num_points = len(pcd.points) if pcd is not None else 0
        formatted_points = f"{num_points:,}".replace(',', '.')
        has_color = hasattr(pcd, 'colors') and len(pcd.colors) > 0

        file_info = (
            f"File: {os.path.basename(self.file_path)}\n"
            f"Points: {formatted_points}\n"
            f"Has Color: {'Yes' if has_color else 'No'}"
        )
        self.update_label_info(file_info)

    def start_downsample_thread(self):
        # Start a new thread for downsampling
        thread = threading.Thread(target=self.downsample_pointcloud)
        thread.start()

    def downsample_pointcloud(self):
        # Retrieve the float parameter from the entry field
        try:
            voxel_size = float(self.entry_field.get())
            if voxel_size <= 0:
                self.show_error("Voxel size must be greater than 0.")
                return
        except ValueError:
            # Check if the user may have entered a float with a comma instead of a period
            if ',' in self.entry_field.get() and '.' not in self.entry_field.get():
                input_text = self.entry_field.get().replace(',', '.')
                self.entry_field.delete(0, 'end')
                self.entry_field.insert(0, input_text)
                self.downsample_pointcloud()
                return
            else:
                self.show_error("Invalid voxel size. Please enter a float value.")
                # Focus on the entry field to allow the user to correct the input
                self.entry_field.focus()
            return

        if self.file_path:
            try:
                # Change the button text to "Loading"
                self.update_button_text(self.btn_downsample, "Loading...")
                # Disable the downsample button and input field during processing
                self.btn_downsample.config(state='disabled')
                self.entry_field.config(state='disabled')

                # Reload the original point cloud data from the file
                original_data = readout_LAS_file(self.file_path)
                if original_data:
                    original_size = len(original_data.points)
                    formatted_original = f"{original_size:,}".replace(',', '.')
                    # Perform downsampling
                    self.pcd_after_dwnsmpl = grid_subsampling(original_data, voxel_size)
                    self.point_cloud_data = grid_subsampling(original_data, voxel_size)
                    num_points = len(self.pcd_after_dwnsmpl.points)
                    formatted_points = f"{num_points:,}".replace(',', '.')
                    # Update the label with the new information
                    file_info = (
                        f"File: {os.path.basename(self.file_path)}\n"
                        f"Points: {formatted_points} (original size: {formatted_original})\n"
                        f"Has Color: {'Yes' if hasattr(self.pcd_after_dwnsmpl, 'colors') and len(self.pcd_after_dwnsmpl.colors) > 0 else 'No'}"  # noqa: E501
                    )
                    self.update_label_info(file_info)
                    # Enable remove noise button
                    self.enable_remove_noise_button()
                    # Enable view point cloud button
                    self.enable_view_point_cloud_button()
                else:
                    self.show_error("Failed to reload the point cloud data.")
            except Exception as e:
                self.handle_exception(e)
            finally:
                # Restore the original button text after processing
                self.update_button_text(self.btn_downsample, self.original_downsample_text)

                # Re-enable the downsample button and input field after processing
                self.btn_downsample.config(state='normal')
                self.entry_field.config(state='normal')

        else:
            self.show_error("No file selected to downsample.")

    def start_remove_noise_thread(self):
        # Start a new thread for noise removal
        thread = threading.Thread(target=self.remove_noise)
        thread.start()

    def remove_noise(self):
        if self.pcd_after_dwnsmpl:
            try:
                # Get the visualization option from the checkbox
                visualize = self.visualize_var.get()

                # Store the original number of points
                original_size = len(self.pcd_after_dwnsmpl.points)

                # Remove noise from the point cloud data with the visualization option
                self.point_cloud_data = remove_noise_statistical(self.pcd_after_dwnsmpl, visualize)

                # Calculate the number of points removed and remaining points
                num_points = len(self.point_cloud_data.points)
                formatted_points = f"{num_points:,}".replace(',', '.')
                points_removed = original_size - num_points
                formatted_removed = f"{points_removed:,}".replace(',', '.')

                # Update the label with the noise reduction status and remaining points
                noise_info = f"Noise removed: {formatted_removed} points\nRemaining points: {formatted_points}"
                self.update_label_remove_noise_info(noise_info)

                # Enable repair and 3D printing buttons
                self.enable_repair_and_3d_printing_buttons()
                # Enable view point cloud button
                self.enable_view_point_cloud_button()
            except Exception as e:
                self.handle_exception(e)
        else:
            self.show_error("No downsampled point cloud data available for noise removal.")

    def start_view_point_cloud_thread(self):
        # Start a new thread for viewing the point cloud
        threading.Thread(target=self.view_point_cloud).start()

    def view_point_cloud(self):
        try:
            if not self.point_cloud_data:
                self.show_error("No point cloud data available for viewing.")
                return

            # Save the point cloud to a temporary PCD file
            temp_pcd_path = "temp_pcd.pcd"
            o3d.io.write_point_cloud(temp_pcd_path, self.point_cloud_data)

            # Launch the open3d_visualization.py script as a separate process
            subprocess.Popen(["python", "Source/open3d_visualization.py", temp_pcd_path])

        except Exception as e:
            self.show_error("An error occurred while previewing the point cloud:\n" + str(e))

    def update_label_info(self, text):
        # Update label text safely from any thread
        self.root.after(0, self.label_info.config, {'text': text})

    def update_label_remove_noise_info(self, text):
        # Update label text safely from any thread
        self.root.after(0, self.label_remove_noise_info.config, {'text': text})

    def enable_remove_noise_button(self):
        # Enable remove noise button safely from any thread
        self.root.after(0, self.btn_remove_noise.config, {'state': 'normal'})

    def enable_view_point_cloud_button(self):
        # Enable view point cloud button safely from any thread
        self.root.after(0, self.btn_view_point_cloud.config, {'state': 'normal'})

    def enable_repair_and_3d_printing_buttons(self):
        # Enable repair and 3D printing buttons safely from any thread
        self.root.after(0, lambda: [
            self.btn_repair_save.config(state='normal'),
            self.btn_open_3d_printing.config(state='normal')
        ])

    def update_button_text(self, button, text):
        # Update button text safely from any thread
        self.root.after(0, button.config, {'text': text})

    def reset(self):
        # Reset the file path, point cloud data, and other variables
        self.file_path = None
        self.point_cloud_data = None
        self.pcd_after_dwnsmpl = None
        self.entry_field.delete(0, 'end')
        self.label_remove_noise_info.config(text="")
        self.disable_buttons()
        self.update_label_info("No file selected")

    def open_3d_printing_module(self):
        if hasattr(self, "printing_window") and self.printing_window.winfo_exists():
            self.printing_window.lift()
        else:
            self.printing_window = tk.Toplevel(self.root)
            PrintingApp(
                self.printing_window,
                point_cloud_data=self.point_cloud_data,
                point_cloud_path=self.file_path)

    def show_error(self, message):
        # Show error message safely from any thread
        self.root.after(0, lambda: messagebox.showerror("Error", message))

    def handle_exception(self, e):
        # Handle exception and show error message
        error_message = f"An error occurred:\n{traceback.format_exc()}"
        self.root.after(0, lambda: messagebox.showerror("Error", error_message))

    def on_close(self):
        # Perform any cleanup or final actions before closing
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()  # Exit the Tkinter main loop
            self.root.destroy()  # Close the Tkinter window
            exit()


# Create the main window and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = PointCloudApp(root)
    root.mainloop()
