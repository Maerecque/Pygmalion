import os
import sys
import tkinter as tk
import threading
import configparser
import pyvista as pv
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.fileHandler import get_save_file_path
from Source.shapeUtils import repair_point_cloud_module
from Source.meshAlterer import mesh_simple_downsample, transform_pcd_to_mesh
from Source.heightMapModule import transform_mesh_to_height_map


class App:
    def __init__(self, root, point_cloud_data=None, point_cloud_path=None):
        self.root = root
        self.root.title("3D Printing Module")
        self.root.resizable(False, False)
        self.root.iconbitmap("Source\\support_files\\logo.ico")

        # Store the point cloud data and path
        self.point_cloud_data = point_cloud_data
        self.point_cloud_path = point_cloud_path

        # Register the validation functions
        self.validate_int = self.root.register(self.validate_integer)
        self.validate_flt = self.root.register(self.validate_float)

        # Validate the data and path
        self.validate_data_and_path()

        # Create widgets
        self.create_widgets()

        # Load presets
        self.load_presets()

    def validate_data_and_path(self):
        if self.point_cloud_data is None:
            raise ValueError("No point cloud data provided.")
        if self.point_cloud_path is None:
            raise ValueError("No point cloud path provided.")
        # Check if the path exists
        if not os.path.exists(self.point_cloud_path):
            raise FileNotFoundError("The provided point cloud path does not exist.")
        return True

    def validate_integer(self, value):
        if value.isdigit() or value == "":
            return True
        return False

    def validate_float(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return value == ""

    def process_point_cloud(self):
        # Disable the Start button to prevent multiple clicks
        self.update_button_state(self.start_button, 'disabled')
        # Update text on the Start button
        self.start_button.config(text="Processing...")

        try:
            # Read point cloud data from the provided data
            pcd = self.point_cloud_data

            # Update start button text
            self.start_button.config(text="Repairing\npoint cloud...")
            # Create a mesh from the point cloud
            stat_mesh = repair_point_cloud_module(
                pcd, visualize=self.visualize_step_var1.get(),
                kdtree_max_nn=int(self.kdtree_nn_entry.get() or 100),
                depth=int(self.depth_entry.get() or 13),
                quantile_value=float(self.quantile_value_entry.get() or 0.01),
                scale=float(self.scale_entry.get() or 2.2)
            )

            # Update start button text
            self.start_button.config(text="Simplifying mesh...")
            # Downsample and simplify the mesh
            simplified_mesh = mesh_simple_downsample(
                stat_mesh,
                pcd,
                float(self.distance_threshold_entry.get() or 0.01),
                self.visualize_step_var2.get()
            )

            # Update start button text
            self.start_button.config(text="Transforming mesh\nto height map...")
            # Transform the mesh into a height map
            floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud = transform_mesh_to_height_map(
                simplified_mesh,
                int(self.gridsize_entry.get() or 100),
                self.visualize_pointcloud_var.get()
            )

            # Update start button text
            self.start_button.config(text="Creating mesh \nfrom pointclouds...")
            # Transform the point clouds into meshes
            floor_plan_volume = transform_pcd_to_mesh(
                floor_plan_point_cloud,
                bool_3d_mesh=self.floor_is_3d_var.get(),
                alpha=float(self.floor_alpha_offset_entry.get() or 0.1),
                tolerance=float(self.floor_tolerance_entry.get() or 0.000001),
                offset=float(self.floor_alpha_offset_entry.get() or 1)
            )
            ceiling_volume = transform_pcd_to_mesh(
                ceiling_point_cloud,
                bool_3d_mesh=self.ceiling_is_3d_var.get(),
                alpha=float(self.ceiling_alpha_offset_entry.get() or 0.2),
                tolerance=float(self.ceiling_tolerance_entry.get() or 0.000001),
                offset=float(self.ceiling_alpha_offset_entry.get() or 1)
            )
            wall_volume = transform_pcd_to_mesh(
                wall_point_cloud,
                bool_3d_mesh=self.walls_is_3d_var.get(),
                alpha=float(self.walls_alpha_offset_entry.get() or 0.225),
                tolerance=float(self.walls_tolerance_entry.get() or 0.000001),
                offset=float(self.walls_alpha_offset_entry.get() or 1)
            )

            # Combine all the parts into one volume
            volume = floor_plan_volume + ceiling_volume + wall_volume

            # Update start button text
            self.start_button.config(text="Fixing remaining holes in volume...")
            # Fix remaining holes in the mesh
            poly_data = volume.extract_geometry()
            filler = vtkFillHolesFilter()
            filler.SetInputData(poly_data)
            filler.Update()
            vtk_volume = filler.GetOutput()

            # Update start button text
            self.start_button.config(text="Saving height map...")
            # Create a filename location for the height map in STL
            export_file_path = get_save_file_path(
                "STL files", ["*.stl"],
                (str(os.path.basename(self.point_cloud_path).split(".")[0]) + ".stl")
            )

            try:
                # Transform the vtk polydata to a pyvista mesh
                output_volume = pv.wrap(vtk_volume)

                # Export the height map as STL
                pv.save_meshio(export_file_path, output_volume)

                # Show success message
                self.show_message("Success", f"Height map saved as {export_file_path}")

            except TypeError:
                self.show_message("Error", "No file save location given.")

        finally:
            # Re-enable the Start button
            self.update_button_state(self.start_button, 'normal')
            # Reset the text on the Start button
            self.start_button.config(text="Start")

    def process_point_cloud_thread(self):
        # Run the processing function in a separate thread
        thread = threading.Thread(target=self.process_point_cloud)
        thread.start()

    def create_widgets(self):
        # Create a main frame
        main_frame = tk.Frame(self.root, padx=5, pady=5)
        main_frame.pack(fill="both")

        # Row 1: Mesh
        mesh_labelframe = tk.LabelFrame(main_frame, text="Mesh")
        mesh_labelframe.pack(fill="x")

        self.visualize_step_var1 = tk.BooleanVar()
        visualize_step_checkbox1 = tk.Checkbutton(mesh_labelframe, text="Visualize step", variable=self.visualize_step_var1)
        visualize_step_checkbox1.pack(side="left")

        input_frame = tk.Frame(mesh_labelframe)
        input_frame.pack(side="right", padx=10, pady=5)

        kdtree_nn_label = tk.Label(input_frame, text="Kdtree NN:", anchor="e", width=12)
        kdtree_nn_label.grid(row=0, column=0)
        self.kdtree_nn_entry = tk.Entry(input_frame, validate="key", validatecommand=(self.validate_int, '%P'))
        self.kdtree_nn_entry.grid(row=0, column=1)

        quantile_value_label = tk.Label(input_frame, text="Quantile Value:", anchor="e", width=12)
        quantile_value_label.grid(row=1, column=0)
        self.quantile_value_entry = tk.Entry(input_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
        self.quantile_value_entry.grid(row=1, column=1)

        depth_label = tk.Label(input_frame, text="Depth:", anchor="e", width=12)
        depth_label.grid(row=0, column=2)
        self.depth_entry = tk.Entry(input_frame, validate="key", validatecommand=(self.validate_int, '%P'))
        self.depth_entry.grid(row=0, column=3)

        scale_label = tk.Label(input_frame, text="Scale:", anchor="e", width=12)
        scale_label.grid(row=1, column=2)
        self.scale_entry = tk.Entry(input_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
        self.scale_entry.grid(row=1, column=3)

        # Row 2: Simplification
        simplification_labelframe = tk.LabelFrame(main_frame, text="Simplification")
        simplification_labelframe.pack(fill="x")

        self.visualize_step_var2 = tk.BooleanVar()
        visualize_step_checkbox2 = tk.Checkbutton(
            simplification_labelframe,
            text="Visualize step",
            variable=self.visualize_step_var2
        )
        visualize_step_checkbox2.pack(side="left")

        input_frame2 = tk.Frame(simplification_labelframe)
        input_frame2.pack(side="right", padx=10, pady=5)

        distance_threshold_label = tk.Label(input_frame2, text="Distance threshold:", anchor="e", width=15)
        distance_threshold_label.grid(row=0, column=0)
        self.distance_threshold_entry = tk.Entry(input_frame2, validate="key", validatecommand=(self.validate_flt, '%P'))
        self.distance_threshold_entry.grid(row=0, column=1)

        # Row 3: Heightmap
        heightmap_labelframe = tk.LabelFrame(main_frame, text="Heightmap")
        heightmap_labelframe.pack(fill="x")

        # Add the checkbox and gridsize entry above the nested labelframes
        checkbox_frame3 = tk.Frame(heightmap_labelframe)
        checkbox_frame3.pack(fill="x")

        self.visualize_pointcloud_var = tk.BooleanVar()
        visualize_pointcloud_checkbox = tk.Checkbutton(
            checkbox_frame3,
            text="Visualize pointcloud of heightmap",
            variable=self.visualize_pointcloud_var
        )
        visualize_pointcloud_checkbox.pack(side="left")

        input_frame3 = tk.Frame(checkbox_frame3)
        input_frame3.pack(side="right", padx=10, pady=5)

        gridsize_label = tk.Label(input_frame3, text="Gridsize:", anchor="e", width=12)
        gridsize_label.grid(row=0, column=0)
        self.gridsize_entry = tk.Entry(input_frame3, validate="key", validatecommand=(self.validate_int, '%P'))
        self.gridsize_entry.grid(row=0, column=1)

        # Adding Ceiling, Walls, and Floor LabelFrames inside Heightmap
        ceiling_labelframe = tk.LabelFrame(heightmap_labelframe, text="Ceiling")
        ceiling_labelframe.pack(fill="x", padx=5, pady=5)

        walls_labelframe = tk.LabelFrame(heightmap_labelframe, text="Walls")
        walls_labelframe.pack(fill="x", padx=5, pady=5)

        floor_labelframe = tk.LabelFrame(heightmap_labelframe, text="Floor")
        floor_labelframe.pack(fill="x", padx=5, pady=5)

        # Add content to Ceiling, Walls, and Floor LabelFrames
        for frame, alpha_offset_entry_name, tolerance_entry_name, threshold_entry_name, is_3d_var_name in [
            (
                ceiling_labelframe,
                'ceiling_alpha_offset_entry',
                'ceiling_tolerance_entry',
                'ceiling_threshold_entry',
                'ceiling_is_3d_var'
            ), (
                walls_labelframe,
                'walls_alpha_offset_entry',
                'walls_tolerance_entry',
                'walls_threshold_entry',
                'walls_is_3d_var'
            ), (
                floor_labelframe,
                'floor_alpha_offset_entry',
                'floor_tolerance_entry',
                'floor_threshold_entry',
                'floor_is_3d_var'
            )
        ]:
            # Left side: Checkboxes
            checkbox_frame = tk.Frame(frame)
            checkbox_frame.pack(side="left", padx=5, pady=5)

            visualize_steps_var = tk.BooleanVar()
            visualize_steps_checkbox = tk.Checkbutton(checkbox_frame, text="Visualize steps", variable=visualize_steps_var)
            visualize_steps_checkbox.pack(anchor="w")

            is_3d_var = tk.BooleanVar()
            is_3d_checkbox = tk.Checkbutton(checkbox_frame, text="Is this a 3D shape?", variable=is_3d_var)
            is_3d_checkbox.pack(anchor="w")

            setattr(self, is_3d_var_name, is_3d_var)

            # Right side: Entries
            entry_frame = tk.Frame(frame)
            entry_frame.pack(side="right", padx=5, pady=5)

            alpha_offset_label = tk.Label(entry_frame, text="Alpha Offset:", anchor="e", width=12)
            alpha_offset_label.grid(row=0, column=0)
            alpha_offset_entry = tk.Entry(entry_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
            alpha_offset_entry.grid(row=0, column=1)
            setattr(self, alpha_offset_entry_name, alpha_offset_entry)

            tolerance_label = tk.Label(entry_frame, text="Tolerance:", anchor="e", width=12)
            tolerance_label.grid(row=1, column=0)
            tolerance_entry = tk.Entry(entry_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
            tolerance_entry.grid(row=1, column=1)
            setattr(self, tolerance_entry_name, tolerance_entry)

            threshold_label = tk.Label(entry_frame, text="Threshold:", anchor="e", width=12)
            threshold_label.grid(row=2, column=0)
            threshold_entry = tk.Entry(entry_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
            threshold_entry.grid(row=2, column=1)
            setattr(self, threshold_entry_name, threshold_entry)

        # Add Buttons at the Bottom
        button_frame = tk.Frame(main_frame, pady=10)
        button_frame.pack(side="bottom")

        back_button = tk.Button(button_frame, text="Back", width=15, height=2, command=self.close_window)
        back_button.pack(side="left", padx=5)

        self.start_button = tk.Button(button_frame, text="Start", width=15, height=2, command=self.process_point_cloud_thread)
        self.start_button.pack(side="left", padx=5)

        # Bind Escape key to close the window
        self.root.bind("<Escape>", lambda event: self.close_window())

    def update_button_state(self, button, state):
        button.config(state=state)

    def show_message(self, title, message):
        message_window = tk.Toplevel(self.root)
        message_window.title(title)
        tk.Label(message_window, text=message, padx=10, pady=10).pack()
        tk.Button(message_window, text="OK", command=message_window.destroy).pack(pady=5)

    def load_presets(self):
        config = configparser.ConfigParser()
        presets_file = 'presets.ini'

        if os.path.exists(presets_file):
            print(f"Loading presets from {presets_file}")
            config.read(presets_file)

            try:
                # Load values into widgets
                self.kdtree_nn_entry.delete(0, tk.END)
                self.kdtree_nn_entry.insert(0, config.get('Settings', 'kdtree_nn', fallback=''))
                self.depth_entry.delete(0, tk.END)
                self.depth_entry.insert(0, config.get('Settings', 'depth', fallback=''))
                self.quantile_value_entry.delete(0, tk.END)
                self.quantile_value_entry.insert(0, config.get('Settings', 'quantile_value', fallback=''))
                self.scale_entry.delete(0, tk.END)
                self.scale_entry.insert(0, config.get('Settings', 'scale', fallback=''))
                self.distance_threshold_entry.delete(0, tk.END)
                self.distance_threshold_entry.insert(0, config.get('Settings', 'distance_threshold', fallback=''))
                self.gridsize_entry.delete(0, tk.END)
                self.gridsize_entry.insert(0, config.get('Settings', 'gridsize', fallback=''))

                # Load heightmap values
                self.floor_alpha_offset_entry.delete(0, tk.END)
                self.floor_alpha_offset_entry.insert(0, config.get('Settings', 'floor_alpha_offset', fallback=''))
                self.floor_tolerance_entry.delete(0, tk.END)
                self.floor_tolerance_entry.insert(0, config.get('Settings', 'floor_tolerance', fallback=''))
                self.floor_threshold_entry.delete(0, tk.END)
                self.floor_threshold_entry.insert(0, config.get('Settings', 'floor_threshold', fallback=''))
                self.ceiling_alpha_offset_entry.delete(0, tk.END)
                self.ceiling_alpha_offset_entry.insert(0, config.get('Settings', 'ceiling_alpha_offset', fallback=''))
                self.ceiling_tolerance_entry.delete(0, tk.END)
                self.ceiling_tolerance_entry.insert(0, config.get('Settings', 'ceiling_tolerance', fallback=''))
                self.ceiling_threshold_entry.delete(0, tk.END)
                self.ceiling_threshold_entry.insert(0, config.get('Settings', 'ceiling_threshold', fallback=''))
                self.walls_alpha_offset_entry.delete(0, tk.END)
                self.walls_alpha_offset_entry.insert(0, config.get('Settings', 'walls_alpha_offset', fallback=''))
                self.walls_tolerance_entry.delete(0, tk.END)
                self.walls_tolerance_entry.insert(0, config.get('Settings', 'walls_tolerance', fallback=''))
                self.walls_threshold_entry.delete(0, tk.END)
                self.walls_threshold_entry.insert(0, config.get('Settings', 'walls_threshold', fallback=''))

                # Load 3D shape booleans
                self.floor_is_3d_var.set(config.getboolean('Settings', 'floor_is_3d', fallback=False))
                self.ceiling_is_3d_var.set(config.getboolean('Settings', 'ceiling_is_3d', fallback=False))
                self.walls_is_3d_var.set(config.getboolean('Settings', 'walls_is_3d', fallback=False))

            except KeyError as e:
                print(f"Missing key in presets file: {e}")
        else:
            print(f"No presets file found at {presets_file}")

    def close_window(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
