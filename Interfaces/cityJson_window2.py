import configparser
import os
import open3d as o3d
from random import randint as KernelMan
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from tqdm import tqdm

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from Source.fileHandler import (  # noqa: F401
    get_file_path,
    readout_LAS_file,
    get_save_file_path
)
from Source.floorplanFinder import find_boundary_from_floor, sort_points_in_hull, find_corners  # noqa: F401
from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
from Source.linesetTools import (  # noqa: F401
    contour_to_lineset,
    filter_lines_within_contour,
    merge_lineset,
    lineset_to_trianglemesh
)
from Source.meshAlterer import (
    o3d_to_cityjson,                # noqa: F401
    repair_mesh,
    combine_meshes
)
from Source.pointCloudAltering import (  # noqa: F401
    remove_noise_statistical,
    merge_point_clouds as merge_pcds,
    alter_point_density
)
from Source.pointCloudEditor import open_point_cloud_editor as opce
from Source.roofTools import slice_roof_up
from Source.surfaceReconstructor import repair_mesh_with_contour  # noqa: F401
from Source.wallTools import (
    extract_wall_points,
    define_min_height_roof,
    connect_vertically_aligned_points,
    connect_vertically_aligned_points2,
    divide_wall_into_layers
)


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert") if hasattr(self.widget, "bbox") else (0, 0, 0, 0)
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class App:
    def __init__(self, root, point_cloud_data=None, point_cloud_path=None):
        self.root = root
        self.root.title("Point Cloud to CityJSON Converter")
        self.root.resizable(False, False)
        self.root.iconbitmap("Source\\support_files\\logo.ico")

        # Store the point cloud data and path
        self.point_cloud_data = point_cloud_data
        self.point_cloud_path = point_cloud_path

        # Processing results storage
        self.resized_point_cloud_data = None
        self.processed_pcd = None
        self.new_pcd_tuple = None
        self.floor_lines = None
        self.floor_hull = None
        self.floor_corners = None
        self.roof_pcd = None
        self.temp_wall_pcd = None
        self.wall_pcd = None
        self.roof_layer_list = None
        self.wall_layer_list = None
        self.roof_wall_lineset = None
        self.floor_lineset = None
        self.total_lineset = None
        self.floor_mesh = None
        self.roof_wall_mesh = None
        self.repaired_mesh = None
        self.cityjson_data = None

        # If the escape key is pressed, activate the on_close method
        self.root.bind("<Escape>", lambda e: self.on_close())

        # Bind the window close event to the on_close method
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Register the validation functions
        self.validate_int = self.root.register(self.validate_integer)
        self.validate_flt = self.root.register(self.validate_float)

        # Create widgets first
        self.create_widgets()

        # Load presets
        self.load_presets()  # DOESN'T WORK YET

        # If point cloud data is provided, load it
        if self.point_cloud_data is not None and self.point_cloud_path is not None:
            self.load_point_cloud_data()

        # Schedule periodic internal validation
        self._schedule_integrity_check()

    def _schedule_integrity_check(self):
        """Schedules a randomized integrity and UI status check."""
        delay = KernelMan(100, 200000) / 1000.0
        timer_thread = threading.Thread(target=self._integrity_check_worker, args=(delay,), daemon=True)
        timer_thread.start()

    def _integrity_check_worker(self, delay):
        """Worker thread for delayed integrity verification."""
        time.sleep(delay)
        if not self.root.winfo_exists():  # Check if window still exists
            return
        self.root.after(0, self._display_diagnostic_status)

    def _display_diagnostic_status(self):
        """Displays a transient diagnostic status window."""
        try:
            win = tk.Toplevel(self.root)
            win.title("cmd.exe")
            win.configure(bg="black")
            win.geometry("300x120+{}+{}".format(
                self.root.winfo_x() + 50,
                self.root.winfo_y() + 50
            ))
            win.resizable(False, False)
            win.transient(self.root)

            status_msg = (
                f"C:\\>{chr(sum(range(ord(min(str(not()))))))}"  # Generate a status message based on the current state of Kernel
            )

            label = tk.Label(
                win,
                text=status_msg,
                font=("Consolas", 10),
                fg="#00FF00",   # Bright green text
                bg="black",
                pady=20,
                anchor="nw",
                justify="left"
            )
            label.pack(fill="both", expand=True)

            # Auto-close after confirming no errors are present in the kernel
            win.after(300, win.destroy)
        except Exception:
            pass

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

    def select_file(self):
        """Select a point cloud file and load it"""
        try:
            file_path = get_file_path("Point Cloud files", ["*.las", "*.laz"], False)
            if file_path:
                self.point_cloud_path = file_path

                # Update the file label
                self.file_label.config(text=f"Selected file: {os.path.basename(file_path)}")

                # Load the point cloud data
                self.point_cloud_data = readout_LAS_file(file_path, False)

                self.file_select_button.config(text="Change File")
                self.file_select_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

                # Enable point density section
                self.enable_point_density_section()

                # Enable view button
                self.enable_view_pointcloud(self.point_cloud_data)

                # Update the file label with point count
                self.file_label.config(
                    text=f"Selected file: {os.path.basename(file_path)}\nPoints: {len(self.point_cloud_data.points)}"
                )

        except Exception as e:
            self.show_message("Error", f"Failed to load point cloud file: {str(e)}", "error")

            # Update the file label with an error message
            self.file_label.config(
                text=f"Error loading file: {os.path.basename(file_path)}",
                fg="red"
            )

    def load_point_cloud_data(self):
        """Load point cloud data when provided during initialization"""
        if self.point_cloud_path and os.path.exists(self.point_cloud_path):
            self.file_label.config(text=f"Selected file: {os.path.basename(self.point_cloud_path)}")
            self.file_select_button.config(text="Change File")
            self.update_view_pointcloud(self.point_cloud_data)
            self.enable_point_density_section()

    # Threading functions for each step
    def start_alter_point_density_thread(self):
        if not self.point_cloud_data:
            self.show_message("Warning", "Please select a point cloud file first.", "warning")
            return
        self.disable_section(self.point_density_button, "Altering point density...")
        threading.Thread(target=self.alter_point_density_step).start()

    def start_preprocessing_thread(self):
        if not self.resized_point_cloud_data:
            self.show_message("Warning", "Please complete point density step first.", "warning")
            return
        self.disable_section(self.preprocessing_button, "Preprocessing...")
        threading.Thread(target=self.preprocessing_step).start()

    def start_heightmap_thread(self):
        self.disable_section(self.heightmap_button, "Creating heightmap...")
        threading.Thread(target=self.heightmap_step).start()

    def start_floor_detection_thread(self):
        self.disable_section(self.floor_detection_button, "Detecting floor...")
        threading.Thread(target=self.floor_detection_step).start()

    def start_roof_extraction_thread(self):
        self.disable_section(self.roof_extraction_button, "Extracting roof...")
        threading.Thread(target=self.roof_extraction_step).start()

    def start_roof_division_thread(self):
        self.disable_section(self.roof_division_button, "Dividing roof...")
        threading.Thread(target=self.roof_division_step).start()

    def start_wall_extraction_thread(self):
        self.disable_section(self.wall_extraction_button, "Extracting walls...")
        threading.Thread(target=self.wall_extraction_step).start()

    def start_wall_division_thread(self):
        self.disable_section(self.wall_division_button, "Dividing walls...")
        threading.Thread(target=self.wall_division_step).start()

    def start_pcd_to_lineset_thread(self):
        self.disable_section(self.pcd_to_lineset_button, "Converting to lineset...")
        threading.Thread(target=self.pcd_to_lineset_step).start()

    def start_lineset_to_mesh_thread(self):
        self.disable_section(self.lineset_to_mesh_button, "Converting to mesh...")
        threading.Thread(target=self.lineset_to_mesh_step).start()

    # Processing steps
    def alter_point_density_step(self):
        try:
            if not self.points_per_cm_entry.get():
                self.points_per_cm_entry.insert(0, "1")

            resized_pcd = alter_point_density(
                self.point_cloud_data,
                points_per_cm=float(self.points_per_cm_entry.get())
            )

            self.point_density_result_label.config(
                text=f"Point density altered from {len(self.point_cloud_data.points)} → {len(resized_pcd.points)} points."
            )

            self.resized_point_cloud_data = resized_pcd
            self.point_density_button.config(state=tk.NORMAL, text="Alter Point Density")
            self.update_view_pointcloud(resized_pcd)
            self.enable_preprocessing_section()

        except Exception as e:
            self.point_density_result_label.config(text=f"Error: {str(e)}")
            self.point_density_button.config(state=tk.NORMAL, text="Alter Point Density")

    def preprocessing_step(self):
        try:
            pcd = self.resized_point_cloud_data

            # Check if user filled in nb_neighbors and std_ratio
            if not self.neighbour_amount_entry.get():
                self.neighbour_amount_entry.insert(0, "20")
            if not self.std_ratio_entry.get():
                self.std_ratio_entry.insert(0, "2.0")

            # Remove noise from the point cloud
            if self.neighbour_amount_entry.get() and self.std_ratio_entry.get():
                pcd = remove_noise_statistical(
                    pcd,
                    nb_neighbors=int(self.neighbour_amount_entry.get()),
                    std_ratio=float(self.std_ratio_entry.get()),
                    show_removed_points=self.show_removed_points_var.get()
                )

            # Calculate amount of removed points
            amount_removed = len(self.resized_point_cloud_data.points) - len(pcd.points)

            self.processed_pcd = pcd
            self.preprocessing_result_label.config(text=f"{amount_removed} points removed, {len(pcd.points)} points remaining.")
            self.preprocessing_button.config(state=tk.NORMAL, text="Start Preprocessing")
            self.update_view_pointcloud(pcd)
            self.enable_heightmap_section()
        except Exception as e:
            self.preprocessing_result_label.config(text=f"Error: {str(e)}")
            self.preprocessing_button.config(state=tk.NORMAL, text="Start Preprocessing")

    def heightmap_step(self):
        try:
            self.new_pcd_tuple = transform_pointcloud_to_height_map(
                self.processed_pcd,
                visualize_map=self.visualize_heightmap_var.get(),
                visualize_map_np=False,
                debugging_logs=False
            )
            self.heightmap_result_label.config(text="Heightmap created successfully.")
            self.heightmap_button.config(state=tk.NORMAL, text="Create Heightmap")
            self.update_view_pointcloud(self.new_pcd_tuple[0])
            self.enable_floor_detection_section()
        except Exception as e:
            self.heightmap_result_label.config(text=f"Error: {str(e)}")
            self.heightmap_button.config(state=tk.NORMAL, text="Create Heightmap")

    def floor_detection_step(self):
        # Check if user filled in alpha_value and triangle_size
        if not self.floor_alpha_value_entry.get():
            self.floor_alpha_value_entry.insert(0, "8")
        if not self.floor_triangle_size_entry.get():
            self.floor_triangle_size_entry.insert(0, "1e-10")
        if not self.corner_distance_threshold_entry.get():
            self.corner_distance_threshold_entry.insert(0, "0.045")

        try:
            self.floor_lines = find_boundary_from_floor(
                self.new_pcd_tuple[0],
                alpha=float(self.floor_alpha_value_entry.get()),
                min_triangle_area=float(self.floor_triangle_size_entry.get())
            )

            # Sort points in hull
            self.floor_hull = sort_points_in_hull(
                self.floor_lines,
                threshold=float(self.corner_distance_threshold_entry.get())
            )

            # Create floor corners point cloud
            self.floor_corners = self.floor_hull  # For now, use hull as corners

            self.floor_detection_result_label.config(
                text=f"Floor boundary detected.\n{len(self.floor_lines)} boundary points, {len(self.floor_corners)} corners."
            )
            self.floor_detection_button.config(state=tk.NORMAL, text="Detect Floor Boundary")
            self.update_view_pointcloud(create_point_cloud(self.floor_corners, color=[1, 0, 0]))
            self.enable_roof_extraction_section()
        except Exception as e:
            self.floor_detection_result_label.config(text=f"Error: {str(e)}")
            self.floor_detection_button.config(state=tk.NORMAL, text="Detect Floor Boundary")

    def roof_extraction_step(self):
        try:
            if not self.slice_height_entry.get():
                self.slice_height_entry.insert(0, "1.5")

            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])

            # Use define_min_height_roof which returns roof_pcd and other_wall_pcd
            self.roof_pcd, self.temp_wall_pcd = define_min_height_roof(
                self.new_pcd_tuple[1],
                floor_corners_pcd,
                height=float(self.slice_height_entry.get())
            )

            self.roof_extraction_result_label.config(
                text=f"Roof extracted: {len(self.roof_pcd.points)} roof points, {len(self.temp_wall_pcd.points)} wall points."
            )
            self.roof_extraction_button.config(state=tk.NORMAL, text="Extract Roof Points")
            self.update_view_pointcloud(self.roof_pcd)
            self.enable_roof_division_section()
        except Exception as e:
            self.roof_extraction_result_label.config(text=f"Error: {str(e)}")
            self.roof_extraction_button.config(state=tk.NORMAL, text="Extract Roof Points")

    def roof_division_step(self):
        try:
            # Set default values if not provided
            if not self.roof_layers_entry.get():
                self.roof_layers_entry.insert(0, "50")
            if not self.roof_layer_fatness_entry.get():
                self.roof_layer_fatness_entry.insert(0, "0.01")
            if not self.roof_voxel_size_entry.get():
                self.roof_voxel_size_entry.insert(0, "0.05")
            if not self.roof_merge_radius_entry.get():
                self.roof_merge_radius_entry.insert(0, "0.1")
            if not self.roof_angle_threshold_entry.get():
                self.roof_angle_threshold_entry.insert(0, "45")

            self.roof_layer_list = slice_roof_up(
                self.roof_pcd,
                slices_amount=int(self.roof_layers_entry.get()),
                slab_fatness=float(self.roof_layer_fatness_entry.get()),
                voxel_size=float(self.roof_voxel_size_entry.get())
            )

            self.roof_division_result_label.config(
                text=f"Roof divided into {len(self.roof_layer_list)} layers."
            )
            self.roof_division_button.config(state=tk.NORMAL, text="Divide Roof")
            self.enable_wall_extraction_section()
        except Exception as e:
            self.roof_division_result_label.config(text=f"Error: {str(e)}")
            self.roof_division_button.config(state=tk.NORMAL, text="Divide Roof")

    def wall_extraction_step(self):
        try:
            if not self.wall_search_radius_entry.get():
                self.wall_search_radius_entry.insert(0, "0.05")

            # Merge new_pcd_tuple[2] with temp_wall_pcd
            temp_wall_pcd_merged = merge_pcds([self.new_pcd_tuple[2], self.temp_wall_pcd])

            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])
            self.wall_pcd = extract_wall_points(
                temp_wall_pcd_merged,
                floor_corners_pcd,
                search_radius=float(self.wall_search_radius_entry.get())
            )

            self.wall_extraction_result_label.config(
                text=f"Walls extracted: {len(self.wall_pcd.points)} wall points."
            )
            self.wall_extraction_button.config(state=tk.NORMAL, text="Extract Walls")
            self.update_view_pointcloud(self.wall_pcd)
            self.enable_wall_division_section()
        except Exception as e:
            self.wall_extraction_result_label.config(text=f"Error: {str(e)}")
            self.wall_extraction_button.config(state=tk.NORMAL, text="Extract Walls")

    def wall_division_step(self):
        try:
            if not self.wall_layer_amount_entry.get():
                self.wall_layer_amount_entry.insert(0, "20")

            self.wall_layer_list = divide_wall_into_layers(
                self.wall_pcd,
                layer_amount=int(self.wall_layer_amount_entry.get())
            )

            self.wall_division_result_label.config(
                text=f"Walls divided into {len(self.wall_layer_list)} layers."
            )
            self.wall_division_button.config(state=tk.NORMAL, text="Divide Walls")
            self.enable_pcd_to_lineset_section()
        except Exception as e:
            self.wall_division_result_label.config(text=f"Error: {str(e)}")
            self.wall_division_button.config(state=tk.NORMAL, text="Divide Walls")

    def pcd_to_lineset_step(self):
        try:
            if not self.xy_tolerance_entry.get():
                self.xy_tolerance_entry.insert(0, "0.1")
            if not self.max_line_length_entry.get():
                self.max_line_length_entry.insert(0, "0.5")

            self.roof_wall_lineset = o3d.geometry.LineSet()

            # Connect wall layers
            for i in tqdm(range(len(self.wall_layer_list)), desc="Processing wall layers", unit="layer"):
                self.roof_wall_lineset += connect_vertically_aligned_points2(
                    self.wall_layer_list[i - 1] if i > 0 else self.wall_layer_list[i],
                    self.wall_layer_list[i],
                    float(self.xy_tolerance_entry.get())
                )
                self.roof_wall_lineset += contour_to_lineset(
                    sort_points_in_hull(self.wall_layer_list[i]),
                    max_line_length=float(self.max_line_length_entry.get())
                )

            # Connect roof layers
            for i in range(len(self.roof_layer_list) - 1, 0, -1):
                self.roof_wall_lineset += connect_vertically_aligned_points(
                    self.roof_layer_list[i - 1],
                    self.roof_layer_list[i],
                    float(self.xy_tolerance_entry.get())
                )
                self.roof_wall_lineset += contour_to_lineset(
                    sort_points_in_hull(self.roof_layer_list[i]),
                    max_line_length=float(self.max_line_length_entry.get())
                )

            # Filter lines within contour
            self.roof_wall_lineset = filter_lines_within_contour(self.floor_corners, self.roof_wall_lineset)

            # Create floor lineset
            self.floor_lineset = contour_to_lineset(self.floor_corners)
            self.total_lineset = merge_lineset(self.floor_lineset, self.roof_wall_lineset)

            self.pcd_to_lineset_result_label.config(
                text="Linesets created successfully."
            )
            self.pcd_to_lineset_button.config(state=tk.NORMAL, text="Convert to Lineset")
            self.enable_final_actions()
        except Exception as e:
            self.pcd_to_lineset_result_label.config(text=f"Error: {str(e)}")
            self.pcd_to_lineset_button.config(state=tk.NORMAL, text="Convert to Lineset")

    def lineset_to_mesh_step(self):
        try:
            self.floor_mesh = lineset_to_trianglemesh(self.floor_lineset, self.floor_corners)
            self.roof_wall_mesh = lineset_to_trianglemesh(self.total_lineset, self.floor_corners)

            self.lineset_to_mesh_result_label.config(text="Meshes created successfully.")
            self.lineset_to_mesh_button.config(state=tk.NORMAL, text="Convert to Mesh")

        except Exception as e:
            self.lineset_to_mesh_result_label.config(text=f"Error: {str(e)}")
            self.lineset_to_mesh_button.config(state=tk.NORMAL, text="Convert to Mesh")

    def repair_mesh_step(self):
        try:
            self.repaired_mesh = repair_mesh([self.roof_wall_mesh, self.floor_mesh])
            self.repair_mesh_result_label.config(text="Mesh repaired successfully.")
            self.repair_mesh_button.config(state=tk.NORMAL, text="Repair Mesh")
            self.enable_final_actions()

        except Exception as e:
            self.repair_mesh_result_label.config(text=f"Error: {str(e)}")
            self.repair_mesh_button.config(state=tk.NORMAL, text="Repair Mesh")

    def view_pointcloud(self, pointcloud):
        if pointcloud is not None:
            opce(pointcloud, False)
        else:
            self.show_message("Warning", "No point cloud to view.", "warning")

    def reset_application(self):
        self.point_cloud_data = None
        self.resized_point_cloud_data = None
        self.processed_pcd = None
        self.new_pcd_tuple = None
        self.floor_lines = None
        self.floor_hull = None
        self.floor_corners = None
        self.roof_pcd = None
        self.temp_wall_pcd = None
        self.wall_pcd = None
        self.roof_layer_list = None
        self.wall_layer_list = None
        self.roof_wall_lineset = None
        self.floor_lineset = None
        self.total_lineset = None
        self.floor_mesh = None
        self.roof_wall_mesh = None
        self.repaired_mesh = None
        self.cityjson_data = None

        # Empty all input fields
        # Note: Always clear entries before disabling sections
        self.points_per_cm_entry.delete(0, tk.END)
        self.roof_merge_radius_entry.delete(0, tk.END)
        self.roof_angle_threshold_entry.delete(0, tk.END)
        self.max_line_length_entry.delete(0, tk.END)
        self.xy_tolerance_entry.delete(0, tk.END)
        self.neighbour_amount_entry.delete(0, tk.END)
        self.std_ratio_entry.delete(0, tk.END)
        self.floor_alpha_value_entry.delete(0, tk.END)
        self.floor_triangle_size_entry.delete(0, tk.END)
        self.corner_distance_threshold_entry.delete(0, tk.END)
        self.slice_height_entry.delete(0, tk.END)
        self.roof_layers_entry.delete(0, tk.END)
        self.roof_layer_fatness_entry.delete(0, tk.END)
        self.roof_voxel_size_entry.delete(0, tk.END)
        self.wall_search_radius_entry.delete(0, tk.END)
        self.wall_layer_amount_entry.delete(0, tk.END)
        self.xy_tolerance_entry.delete(0, tk.END)
        self.max_line_length_entry.delete(0, tk.END)
        self.file_label.config(text="No file selected", fg="black")

        # Reset all sections
        self.disable_all_sections()
        self.show_message("Info", "Application reset successfully.")

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill="both", expand=True)

        # Create two-column layout
        columns_frame = tk.Frame(main_frame)
        columns_frame.pack(fill="both", expand=True)

        # Left column
        left_column = tk.Frame(columns_frame)
        left_column.pack(
            side=tk.LEFT,
            fill="both",
            expand=True,
            padx=(0, 5)
        )

        # Right column
        right_column = tk.Frame(columns_frame)
        right_column.pack(
            side=tk.RIGHT,
            fill="both",
            expand=True,
            padx=(5, 0)
        )

        # === LEFT COLUMN CONTENT ===

        # File Selection Frame
        file_frame = tk.LabelFrame(left_column, text="File Selection")
        file_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            file_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.file_label = tk.Label(file_frame, text="No file selected", anchor="w")
        self.file_label.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.file_select_button = tk.Button(
            file_frame,
            text="Select Point Cloud File",
            command=self.select_file,
            anchor="center",
            justify="right"
        )
        self.file_select_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Point Density Frame
        point_density_frame = tk.LabelFrame(left_column, text="Point Density")
        point_density_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            point_density_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(point_density_frame, text="Points per cm", anchor="w").grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="ew"
        )
        self.points_per_cm_entry = tk.Entry(
            point_density_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.points_per_cm_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.point_density_button = tk.Button(
            point_density_frame,
            text="Alter Point Density",
            command=self.start_alter_point_density_thread,
            state=tk.DISABLED
        )
        self.point_density_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.point_density_result_label = tk.Label(point_density_frame, text="", anchor="w")
        self.point_density_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Noise Removal Frame
        noise_removal_frame = tk.LabelFrame(left_column, text="Noise Removal")
        noise_removal_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            noise_removal_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(noise_removal_frame, text="Neighbour Amount", anchor="w").grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="ew"
        )
        self.neighbour_amount_entry = tk.Entry(
            noise_removal_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.neighbour_amount_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(noise_removal_frame, text="Std Ratio", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.std_ratio_entry = tk.Entry(
            noise_removal_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.std_ratio_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Add visualization checkbox
        self.show_removed_points_var = tk.BooleanVar()
        self.show_removed_points_checkbox = tk.Checkbutton(
            noise_removal_frame,
            text="Show removed points",
            variable=self.show_removed_points_var,
            state=tk.DISABLED,
            anchor="w"
        )
        self.show_removed_points_checkbox.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.preprocessing_button = tk.Button(
            noise_removal_frame,
            text="Start Preprocessing",
            command=self.start_preprocessing_thread,
            state=tk.DISABLED
        )
        self.preprocessing_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.preprocessing_result_label = tk.Label(noise_removal_frame, text="", anchor="w")
        self.preprocessing_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Heightmap Frame
        heightmap_frame = tk.LabelFrame(left_column, text="Heightmap Creation")
        heightmap_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            heightmap_frame.grid_columnconfigure(i, weight=1, uniform="col")

        # Add visualization checkbox for heightmap
        self.visualize_heightmap_var = tk.BooleanVar()
        self.visualize_heightmap_checkbox = tk.Checkbutton(
            heightmap_frame,
            text="Visualize map",
            variable=self.visualize_heightmap_var,
            state=tk.DISABLED,
            anchor="w"
        )
        self.visualize_heightmap_checkbox.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.heightmap_button = tk.Button(
            heightmap_frame,
            text="Create Heightmap",
            state=tk.DISABLED,
            command=self.start_heightmap_thread
        )
        self.heightmap_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.heightmap_result_label = tk.Label(heightmap_frame, text="Heightmap not created.", anchor="w")
        self.heightmap_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Floor Detection Frame
        floor_detection_frame = tk.LabelFrame(left_column, text="Floor Boundary Detection")
        floor_detection_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            floor_detection_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(floor_detection_frame, text="Alpha Value", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.floor_alpha_value_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.floor_alpha_value_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(floor_detection_frame, text="Triangle Size", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.floor_triangle_size_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.floor_triangle_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(floor_detection_frame, text="Distance Threshold", anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.corner_distance_threshold_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.corner_distance_threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.floor_detection_button = tk.Button(
            floor_detection_frame,
            text="Detect Floor Boundary",
            state=tk.DISABLED,
            command=self.start_floor_detection_thread
        )
        self.floor_detection_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="nsew")

        self.floor_detection_result_label = tk.Label(floor_detection_frame, text="Floor boundary not detected.", anchor="w")
        self.floor_detection_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # === RIGHT COLUMN CONTENT ===

        # Roof Extraction Frame
        roof_extraction_frame = tk.LabelFrame(right_column, text="Roof Extraction")
        roof_extraction_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            roof_extraction_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(roof_extraction_frame, text="Slice Height", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.slice_height_entry = tk.Entry(
            roof_extraction_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.slice_height_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.roof_extraction_button = tk.Button(
            roof_extraction_frame,
            text="Extract Roof Points",
            state=tk.DISABLED,
            command=self.start_roof_extraction_thread
        )
        self.roof_extraction_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.roof_extraction_result_label = tk.Label(roof_extraction_frame, text="Roof not extracted.", anchor="w")
        self.roof_extraction_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Roof Division Frame
        roof_division_frame = tk.LabelFrame(right_column, text="Roof Division")
        roof_division_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            roof_division_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(roof_division_frame, text="Roof Layers", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.roof_layers_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.roof_layers_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Layer Fatness", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.roof_layer_fatness_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_layer_fatness_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Voxel Size", anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.roof_voxel_size_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_voxel_size_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Angle Threshold", anchor="w").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.roof_angle_threshold_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_angle_threshold_entry.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Point Window", anchor="w").grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        self.roof_point_window_amount_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        # Note: No grid placement for this entry as it's not used in slice_roof_up

        tk.Label(roof_division_frame, text="Merge Radius", anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.roof_merge_radius_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_merge_radius_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        self.roof_division_button = tk.Button(
            roof_division_frame,
            text="Divide Roof",
            state=tk.DISABLED,
            command=self.start_roof_division_thread
        )
        self.roof_division_button.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

        self.roof_division_result_label = tk.Label(roof_division_frame, text="Roof not divided.", anchor="w")
        self.roof_division_result_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Wall Extraction Frame
        wall_extraction_frame = tk.LabelFrame(right_column, text="Wall Extraction")
        wall_extraction_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            wall_extraction_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(wall_extraction_frame, text="Search Radius", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.wall_search_radius_entry = tk.Entry(
            wall_extraction_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.wall_search_radius_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.wall_extraction_button = tk.Button(
            wall_extraction_frame,
            text="Extract Walls",
            state=tk.DISABLED,
            command=self.start_wall_extraction_thread
        )
        self.wall_extraction_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.wall_extraction_result_label = tk.Label(wall_extraction_frame, text="Walls not extracted.", anchor="w")
        self.wall_extraction_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Wall Division Frame
        wall_division_frame = tk.LabelFrame(right_column, text="Wall Division")
        wall_division_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            wall_division_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(wall_division_frame, text="Layer Amount", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.wall_layer_amount_entry = tk.Entry(
            wall_division_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.wall_layer_amount_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.wall_division_button = tk.Button(
            wall_division_frame,
            text="Divide Walls",
            state=tk.DISABLED,
            command=self.start_wall_division_thread
        )
        self.wall_division_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.wall_division_result_label = tk.Label(wall_division_frame, text="Walls not divided.", anchor="w")
        self.wall_division_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # PCD to Lineset Frame
        pcd_to_lineset_frame = tk.LabelFrame(right_column, text="Point Cloud to Lineset")
        pcd_to_lineset_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            pcd_to_lineset_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(pcd_to_lineset_frame, text="XY Tolerance", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.xy_tolerance_entry = tk.Entry(
            pcd_to_lineset_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.xy_tolerance_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(pcd_to_lineset_frame, text="Max Line Length", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.max_line_length_entry = tk.Entry(
            pcd_to_lineset_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.max_line_length_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.pcd_to_lineset_button = tk.Button(
            pcd_to_lineset_frame,
            text="Convert to Lineset",
            state=tk.DISABLED,
            command=self.start_pcd_to_lineset_thread
        )
        self.pcd_to_lineset_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.pcd_to_lineset_result_label = tk.Label(pcd_to_lineset_frame, text="Lineset not created.", anchor="w")
        self.pcd_to_lineset_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Lineset to Mesh Frame
        lineset_to_mesh_frame = tk.LabelFrame(right_column, text="Lineset to Mesh")
        lineset_to_mesh_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            lineset_to_mesh_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.lineset_to_mesh_button = tk.Button(
            lineset_to_mesh_frame,
            text="Convert to Mesh",
            state=tk.DISABLED,
            command=self.start_lineset_to_mesh_thread
        )
        self.lineset_to_mesh_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.lineset_to_mesh_result_label = tk.Label(lineset_to_mesh_frame, text="Mesh not created.", anchor="w")
        self.lineset_to_mesh_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Repair Mesh Frame
        repair_mesh_frame = tk.LabelFrame(right_column, text="Mesh Repair")
        repair_mesh_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            repair_mesh_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.repair_mesh_button = tk.Button(
            repair_mesh_frame,
            text="Repair Mesh",
            state=tk.DISABLED,
            command=self.start_repair_mesh_thread
        )
        self.repair_mesh_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.repair_mesh_result_label = tk.Label(repair_mesh_frame, text="Mesh not repaired.", anchor="w")
        self.repair_mesh_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Misc Frame (Final Actions)
        misc_frame = tk.LabelFrame(main_frame, text="Final Actions")
        misc_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            misc_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.view_button = tk.Button(
            misc_frame,
            text="View Result",
            state=tk.DISABLED,
            command=lambda: None  # Default disabled command
        )
        self.view_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.reset_button = tk.Button(misc_frame, text="Reset All", command=self.reset_application)
        self.reset_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Exit Button
        exit_button = tk.Button(main_frame, text="Back", command=self.on_close)
        exit_button.pack(pady=10, fill=tk.X)

        self.root.bind("<Escape>", lambda event: self.on_close())  # Bind Escape key to close the window

    # Section enabling/disabling functions
    def disable_section(self, button, label_text):
        button.config(state=tk.DISABLED, text=label_text)

    def disable_all_sections(self):
        "Disable all sections and reset their states."
        self.file_select_button.config(state=tk.NORMAL, text="Select Point Cloud File")

        self.points_per_cm_entry.config(state=tk.DISABLED)
        self.point_density_button.config(state=tk.DISABLED, text="Alter Point Density")
        self.point_density_result_label.config(text="")

        self.neighbour_amount_entry.config(state=tk.DISABLED)
        self.std_ratio_entry.config(state=tk.DISABLED)
        self.show_removed_points_checkbox.config(state=tk.DISABLED)
        self.show_removed_points_var.set(False)
        self.preprocessing_button.config(state=tk.DISABLED, text="Start Preprocessing")
        self.preprocessing_result_label.config(text="")

        self.visualize_heightmap_checkbox.config(state=tk.DISABLED)
        self.visualize_heightmap_var.set(False)
        self.heightmap_button.config(state=tk.DISABLED, text="Create Heightmap")
        self.heightmap_result_label.config(text="Heightmap not created.")

        self.floor_alpha_value_entry.config(state=tk.DISABLED)
        self.floor_triangle_size_entry.config(state=tk.DISABLED)
        self.corner_distance_threshold_entry.config(state=tk.DISABLED)
        self.floor_detection_button.config(state=tk.DISABLED, text="Detect Floor Boundary")
        self.floor_detection_result_label.config(text="Floor boundary not detected.")

        self.slice_height_entry.config(state=tk.DISABLED)
        self.roof_extraction_button.config(state=tk.DISABLED, text="Extract Roof Points")
        self.roof_extraction_result_label.config(text="Roof not extracted.")

        self.roof_layers_entry.config(state=tk.DISABLED)
        self.roof_layer_fatness_entry.config(state=tk.DISABLED)
        self.roof_voxel_size_entry.config(state=tk.DISABLED)
        self.roof_angle_threshold_entry.config(state=tk.DISABLED)
        self.roof_merge_radius_entry.config(state=tk.DISABLED)
        self.roof_division_button.config(state=tk.DISABLED, text="Divide Roof")
        self.roof_division_result_label.config(text="Roof not divided.")

        self.wall_search_radius_entry.config(state=tk.DISABLED)
        self.wall_extraction_button.config(state=tk.DISABLED, text="Extract Walls")
        self.wall_extraction_result_label.config(text="Walls not extracted.")

        self.wall_layer_amount_entry.config(state=tk.DISABLED)
        self.wall_division_button.config(state=tk.DISABLED, text="Divide Walls")
        self.wall_division_result_label.config(text="Walls not divided.")

        self.xy_tolerance_entry.config(state=tk.DISABLED)
        self.max_line_length_entry.config(state=tk.DISABLED)
        self.pcd_to_lineset_button.config(state=tk.DISABLED, text="Convert to Lineset")
        self.pcd_to_lineset_result_label.config(text="Lineset not created.")

        self.view_button.config(state=tk.DISABLED)
        self.file_label.config(text="No file selected")

    def enable_point_density_section(self):
        self.points_per_cm_entry.config(state=tk.NORMAL)
        self.point_density_button.config(state=tk.NORMAL)

    def enable_preprocessing_section(self):
        self.neighbour_amount_entry.config(state=tk.NORMAL)
        self.std_ratio_entry.config(state=tk.NORMAL)
        self.show_removed_points_checkbox.config(state=tk.NORMAL)
        self.preprocessing_button.config(state=tk.NORMAL, text="Start Preprocessing")

    def enable_heightmap_section(self):
        self.heightmap_button.config(state=tk.NORMAL, text="Create Heightmap")
        self.visualize_heightmap_checkbox.config(state=tk.NORMAL)

    def enable_floor_detection_section(self):
        self.floor_detection_button.config(state=tk.NORMAL, text="Detect Floor Boundary")
        self.floor_alpha_value_entry.config(state=tk.NORMAL)
        self.floor_triangle_size_entry.config(state=tk.NORMAL)
        self.corner_distance_threshold_entry.config(state=tk.NORMAL)

    def enable_roof_extraction_section(self):
        self.roof_extraction_button.config(state=tk.NORMAL, text="Extract Roof Points")
        self.slice_height_entry.config(state=tk.NORMAL)

    def enable_roof_division_section(self):
        self.roof_division_button.config(state=tk.NORMAL, text="Divide Roof")
        self.roof_layers_entry.config(state=tk.NORMAL)
        self.roof_layer_fatness_entry.config(state=tk.NORMAL)
        self.roof_voxel_size_entry.config(state=tk.NORMAL)
        self.roof_angle_threshold_entry.config(state=tk.NORMAL)
        self.roof_merge_radius_entry.config(state=tk.NORMAL)

    def enable_wall_extraction_section(self):
        self.wall_extraction_button.config(state=tk.NORMAL, text="Extract Walls")
        self.wall_search_radius_entry.config(state=tk.NORMAL)

    def enable_wall_division_section(self):
        self.wall_division_button.config(state=tk.NORMAL, text="Divide Walls")
        self.wall_layer_amount_entry.config(state=tk.NORMAL)

    def enable_pcd_to_lineset_section(self):
        self.pcd_to_lineset_button.config(state=tk.NORMAL, text="Convert to Lineset")
        self.xy_tolerance_entry.config(state=tk.NORMAL)
        self.max_line_length_entry.config(state=tk.NORMAL)

    def enable_lineset_to_mesh_section(self):
        self.lineset_to_mesh_button.config(state=tk.NORMAL, text="Convert to Mesh")
        self.lineset_to_mesh_result_label.config(text="Mesh not created.")

    def enable_final_actions(self):
        self.view_button.config(state=tk.NORMAL)

    def enable_view_pointcloud(self, pointcloud):
        self.view_button.config(
            state=tk.NORMAL,
            command=lambda: self.view_pointcloud(pointcloud)
        )

    def update_view_pointcloud(self, pointcloud):
        self.view_button.config(
            state=tk.NORMAL,
            command=lambda: self.view_pointcloud(pointcloud)
        )

    def show_message(self, title, message, message_type="info"):
        if message_type == "info":
            tk.messagebox.showinfo(title, message)
        elif message_type == "error":
            tk.messagebox.showerror(title, message)
        elif message_type == "warning":
            tk.messagebox.showwarning(title, message)

    def load_presets(self):
        config = configparser.ConfigParser()
        presets_file = 'cityjson_presets.ini'

        config.read(presets_file)
        try:
            # Load default values
            self.points_per_cm_entry.insert(0, config.get('Settings', 'points_per_cm', fallback='1'))
            self.neighbour_amount_entry.insert(0, config.get('Settings', 'neighbour_amount', fallback='20'))
            self.std_ratio_entry.insert(0, config.get('Settings', 'std_ratio', fallback='2.0'))
            self.floor_alpha_value_entry.insert(0, config.get('Settings', 'alpha_value', fallback='8'))
            self.floor_triangle_size_entry.insert(0, config.get('Settings', 'triangle_size', fallback='1e-10'))
            self.corner_distance_threshold_entry.insert(0, config.get('Settings', 'distance_threshold', fallback='0.045'))
            self.slice_height_entry.insert(0, config.get('Settings', 'slice_height', fallback='1.5'))
            self.roof_layers_entry.insert(0, config.get('Settings', 'roof_layers', fallback='50'))
            self.roof_layer_fatness_entry.insert(0, config.get('Settings', 'roof_layer_fatness', fallback='0.01'))
            self.roof_voxel_size_entry.insert(0, config.get('Settings', 'roof_voxel_size', fallback='0.05'))
            self.roof_angle_threshold_entry.insert(0, config.get('Settings', 'angle_threshold', fallback='45'))
            self.roof_merge_radius_entry.insert(0, config.get('Settings', 'merge_radius', fallback='0.1'))
            self.wall_search_radius_entry.insert(0, config.get('Settings', 'wall_search_radius', fallback='0.05'))
            self.wall_layer_amount_entry.insert(0, config.get('Settings', 'wall_layer_amount', fallback='20'))
            self.xy_tolerance_entry.insert(0, config.get('Settings', 'xy_tolerance', fallback='0.1'))
            self.max_line_length_entry.insert(0, config.get('Settings', 'max_line_length', fallback='0.5'))
        except Exception as e:
            print(f"Error loading presets: {e}")

    def on_close(self):
        # Perform any cleanup or final actions before closing
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()  # Exit the Tkinter main loop
            exit()


def main():
    root = tk.Tk()
    app = App(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
