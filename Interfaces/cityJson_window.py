import os
import sys
import tkinter as tk
from tkinter import ttk
import threading
import configparser

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.fileHandler import get_file_path, readout_LAS_file, get_save_file_path  # noqa: F401
from Source.pointCloudAltering import remove_noise_statistical as rns
from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
from Source.pointCloudEditor import open_point_cloud_editor as opce
from Source.pointCloudAltering import merge_point_clouds as merge_pcds
from Source.cityJsonModule import (
    find_lines_in_pointcloud, sort_points_in_hull, find_corners,
    create_correct_height_slice, keep_wall_points_from_x_height,
    slice_roof_up, keep_highest_point_above_corner
)


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

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
        self.processed_pcd = None
        self.new_pcd_tuple = None
        self.floor_lines = None
        self.floor_hull = None
        self.floor_corners = None
        self.wall_slice = None
        self.new_roof_pcd = None
        self.sliced_roof = None
        self.filtered_sliced_roof = None

        # If no point cloud data or path is provided, let the user select a file
        if self.point_cloud_data is None or self.point_cloud_path is None:
            self.point_cloud_path = get_file_path(
                "Point Cloud files", ["*.las", "*.laz"]
            )
            if not self.point_cloud_path:
                raise ValueError("No point cloud file selected.")
            self.point_cloud_data = readout_LAS_file(self.point_cloud_path)

        # Register the validation functions
        self.validate_int = self.root.register(self.validate_integer)
        self.validate_flt = self.root.register(self.validate_float)

        # Button width for consistency
        self.button_width = 20

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

    # Threading functions for each step
    def start_preprocessing_thread(self):
        self.disable_section(self.preprocessing_button, "Preprocessing...")
        threading.Thread(target=self.preprocessing_step).start()

    def start_heightmap_thread(self):
        self.disable_section(self.heightmap_button, "Creating heightmap...")
        threading.Thread(target=self.heightmap_step).start()

    def start_floor_detection_thread(self):
        self.disable_section(self.floor_detection_button, "Detecting floor...")
        threading.Thread(target=self.floor_detection_step).start()

    def start_corner_detection_thread(self):
        self.disable_section(self.corner_detection_button, "Detecting corners...")
        threading.Thread(target=self.corner_detection_step).start()

    def start_wall_slice_thread(self):
        self.disable_section(self.wall_slice_button, "Creating wall slice...")
        threading.Thread(target=self.wall_slice_step).start()

    def start_roof_extraction_thread(self):
        self.disable_section(self.roof_extraction_button, "Extracting roof...")
        threading.Thread(target=self.roof_extraction_step).start()

    def start_roof_slice_thread(self):
        self.disable_section(self.roof_slice_button, "Slicing roof...")
        threading.Thread(target=self.roof_slice_step).start()

    def start_roof_outline_thread(self):
        self.disable_section(self.roof_outline_button, "Finding roof outline...")
        threading.Thread(target=self.roof_outline_step).start()

    def start_combine_results_thread(self):
        self.disable_section(self.combine_results_button, "Combining results...")
        threading.Thread(target=self.combine_results_step).start()

    # Processing steps
    def preprocessing_step(self):
        try:
            pcd = self.point_cloud_data

            # Remove noise from the point cloud
            if self.neighbour_amount_entry.get() and self.std_ratio_entry.get():
                pcd = rns(
                    pcd,
                    nb_neighbors=int(self.neighbour_amount_entry.get() or 20),
                    std_ratio=float(self.std_ratio_entry.get() or 2.0)
                )

            # Downsample if specified
            if self.voxel_size_entry.get():
                pcd = pcd.voxel_down_sample(voxel_size=float(self.voxel_size_entry.get()))

            self.processed_pcd = pcd
            self.preprocessing_result_label.config(text=f"Preprocessing completed.\n{len(pcd.points)} points remaining.")
            self.preprocessing_button.config(state=tk.NORMAL, text="Start Preprocessing")
            self.enable_heightmap_section()
        except Exception as e:
            self.preprocessing_result_label.config(text=f"Error: {str(e)}")
            self.preprocessing_button.config(state=tk.NORMAL, text="Start Preprocessing")

    def heightmap_step(self):
        try:
            self.new_pcd_tuple = transform_pointcloud_to_height_map(
                self.processed_pcd,
                grid_size=int(self.grid_size_entry.get() or 500),
                visualize_map=False,
                debugging_logs=False
            )
            self.heightmap_result_label.config(text="Heightmap created successfully.")
            self.heightmap_button.config(state=tk.NORMAL, text="Create Heightmap")
            self.enable_floor_detection_section()
        except Exception as e:
            self.heightmap_result_label.config(text=f"Error: {str(e)}")
            self.heightmap_button.config(state=tk.NORMAL, text="Create Heightmap")

    def floor_detection_step(self):
        try:
            self.floor_lines = find_lines_in_pointcloud(
                self.new_pcd_tuple[0],
                alpha=float(self.alpha_value_entry.get() or 11),
                min_triangle_area=float(self.triangle_size_entry.get() or 1e-10)
            )
            self.floor_detection_result_label.config(text=f"Floor boundary detected.\n{len(self.floor_lines)} boundary points.")
            self.floor_detection_button.config(state=tk.NORMAL, text="Detect Floor Boundary")
            self.enable_corner_detection_section()
        except Exception as e:
            self.floor_detection_result_label.config(text=f"Error: {str(e)}")
            self.floor_detection_button.config(state=tk.NORMAL, text="Detect Floor Boundary")

    def corner_detection_step(self):
        try:
            self.floor_hull = sort_points_in_hull(
                self.floor_lines,
                threshold=float(self.distance_threshold_entry.get() or 0.05)
            )
            self.floor_corners = find_corners(
                self.floor_hull,
                angle_threshold_deg=float(self.angle_threshold_entry.get() or 45),
                window=2,
                merge_radius=int(self.merge_radius_entry.get() or 1)
            )
            self.corner_detection_result_label.config(text=f"Corners detected.\n{len(self.floor_corners)} corners found.")
            self.corner_detection_button.config(state=tk.NORMAL, text="Detect Corners")
            self.enable_wall_slice_section()
        except Exception as e:
            self.corner_detection_result_label.config(text=f"Error: {str(e)}")
            self.corner_detection_button.config(state=tk.NORMAL, text="Detect Corners")

    def wall_slice_step(self):
        try:
            self.wall_slice = create_correct_height_slice(
                self.new_pcd_tuple[1],
                create_point_cloud(self.floor_corners, color=[1, 0, 0]),
                height=float(self.slice_height_entry.get() or 1.5),
                search_radius=float(self.search_radius_entry.get() or 0.025)
            )
            self.wall_slice_result_label.config(text=f"Wall slice created.\n{len(self.wall_slice.points)} wall points.")
            self.wall_slice_button.config(state=tk.NORMAL, text="Create Wall Slice")
            self.enable_roof_extraction_section()
        except Exception as e:
            self.wall_slice_result_label.config(text=f"Error: {str(e)}")
            self.wall_slice_button.config(state=tk.NORMAL, text="Create Wall Slice")

    def roof_extraction_step(self):
        try:
            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])
            self.new_roof_pcd = keep_wall_points_from_x_height(
                self.new_pcd_tuple[1],
                floor_corners_pcd,
                height=float(self.slice_height_entry.get() or 1.5)
            )
            self.roof_extraction_result_label.config(text=f"Roof points extracted.\n{len(self.new_roof_pcd.points)} roof points.")
            self.roof_extraction_button.config(state=tk.NORMAL, text="Extract Roof Points")
            self.enable_roof_slice_section()
        except Exception as e:
            self.roof_extraction_result_label.config(text=f"Error: {str(e)}")
            self.roof_extraction_button.config(state=tk.NORMAL, text="Extract Roof Points")

    def roof_slice_step(self):
        try:
            self.sliced_roof = slice_roof_up(
                self.new_roof_pcd,
                slices_amount=int(self.roof_layers_entry.get() or 5),
                slab_fatness=float(self.layer_fatness_entry.get() or 0.0075)
            )
            self.roof_slice_result_label.config(text=f"Roof sliced.\n{len(self.sliced_roof.points)} sliced points.")
            self.roof_slice_button.config(state=tk.NORMAL, text="Slice Roof")
            self.enable_roof_outline_section()
        except Exception as e:
            self.roof_slice_result_label.config(text=f"Error: {str(e)}")
            self.roof_slice_button.config(state=tk.NORMAL, text="Slice Roof")

    def roof_outline_step(self):
        try:
            floor_hull_pcd = create_point_cloud(self.floor_hull)
            self.filtered_sliced_roof = keep_highest_point_above_corner(
                floor_hull_pcd,
                self.sliced_roof,
                search_radius=float(self.roof_search_radius_entry.get() or 0.025)
            )
            self.roof_outline_result_label.config(
                text=f"Roof outline found.\n{len(self.filtered_sliced_roof.points)} outline points."
            )
            self.roof_outline_button.config(state=tk.NORMAL, text="Find Roof Outline")
            self.enable_combine_results_section()
        except Exception as e:
            self.roof_outline_result_label.config(text=f"Error: {str(e)}")
            self.roof_outline_button.config(state=tk.NORMAL, text="Find Roof Outline")

    def combine_results_step(self):
        try:
            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])
            wall_floor_merge = merge_pcds([floor_corners_pcd, self.wall_slice])
            self.processed_pcd = merge_pcds([wall_floor_merge, self.filtered_sliced_roof])

            self.combine_results_result_label.config(text=f"Results combined.\n{len(self.processed_pcd.points)} total points.")
            self.combine_results_button.config(state=tk.NORMAL, text="Combine Results")
            self.enable_final_actions()
        except Exception as e:
            self.combine_results_result_label.config(text=f"Error: {str(e)}")
            self.combine_results_button.config(state=tk.NORMAL, text="Combine Results")

    def view_pointcloud(self):
        if self.processed_pcd is not None:
            opce(self.processed_pcd)
        else:
            self.show_message("Warning", "No processed point cloud to view. Please complete processing first.", "warning")

    def reset_application(self):
        self.processed_pcd = None
        self.new_pcd_tuple = None
        self.floor_lines = None
        self.floor_hull = None
        self.floor_corners = None
        self.wall_slice = None
        self.new_roof_pcd = None
        self.sliced_roof = None
        self.filtered_sliced_roof = None

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
        left_column.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 5))

        # Right column
        right_column = tk.Frame(columns_frame)
        right_column.pack(side=tk.RIGHT, fill="both", expand=True, padx=(5, 0))

        # === LEFT COLUMN CONTENT ===

        # File Selection Frame
        file_frame = ttk.LabelFrame(left_column, text="File Selection", padding=10)
        file_frame.pack(fill="x", pady=5)
        ttk.Label(file_frame, text=f"Selected file: {os.path.basename(self.point_cloud_path)}").pack(anchor="w")

        # Downsampling Frame
        downsampling_frame = tk.LabelFrame(left_column, text="Downsampling")
        downsampling_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            downsampling_frame.grid_columnconfigure(i, weight=1)

        tk.Label(downsampling_frame, text="Voxel Size").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.voxel_size_entry = tk.Entry(downsampling_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
        self.voxel_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Noise Removal Frame
        noise_removal_frame = tk.LabelFrame(left_column, text="Noise Removal")
        noise_removal_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            noise_removal_frame.grid_columnconfigure(i, weight=1)

        tk.Label(noise_removal_frame, text="Neighbour Amount").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.neighbour_amount_entry = tk.Entry(noise_removal_frame, validate="key", validatecommand=(self.validate_int, '%P'))
        self.neighbour_amount_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(noise_removal_frame, text="Std Ratio").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.std_ratio_entry = tk.Entry(noise_removal_frame, validate="key", validatecommand=(self.validate_flt, '%P'))
        self.std_ratio_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.preprocessing_button = tk.Button(
            noise_removal_frame,
            text="Start Preprocessing",
            width=self.button_width,
            command=self.start_preprocessing_thread
        )
        self.preprocessing_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.preprocessing_result_label = tk.Label(noise_removal_frame, text="Preprocessing not started.", anchor="w")
        self.preprocessing_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Heightmap Frame
        heightmap_frame = tk.LabelFrame(left_column, text="Heightmap Creation")
        heightmap_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            heightmap_frame.grid_columnconfigure(i, weight=1)

        tk.Label(heightmap_frame, text="Grid Size").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.grid_size_entry = tk.Entry(
            heightmap_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.grid_size_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.heightmap_button = tk.Button(
            heightmap_frame,
            text="Create Heightmap",
            width=self.button_width,
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
            floor_detection_frame.grid_columnconfigure(i, weight=1)

        tk.Label(floor_detection_frame, text="Alpha Value").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.alpha_value_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.alpha_value_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(floor_detection_frame, text="Triangle Size").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.triangle_size_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.triangle_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.floor_detection_button = tk.Button(
            floor_detection_frame,
            text="Detect Floor Boundary",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_floor_detection_thread
        )
        self.floor_detection_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.floor_detection_result_label = tk.Label(floor_detection_frame, text="Floor boundary not detected.", anchor="w")
        self.floor_detection_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # === RIGHT COLUMN CONTENT ===

        # Corner Detection Frame
        corner_detection_frame = tk.LabelFrame(right_column, text="Corner Detection")
        corner_detection_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            corner_detection_frame.grid_columnconfigure(i, weight=1)

        tk.Label(corner_detection_frame, text="Distance Threshold").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.distance_threshold_entry = tk.Entry(
            corner_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.distance_threshold_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(corner_detection_frame, text="Angle Threshold").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.angle_threshold_entry = tk.Entry(
            corner_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.angle_threshold_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(corner_detection_frame, text="Merge Radius").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.merge_radius_entry = tk.Entry(
            corner_detection_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.merge_radius_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.corner_detection_button = tk.Button(
            corner_detection_frame,
            text="Detect Corners",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_corner_detection_thread
        )
        self.corner_detection_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="nsew")

        self.corner_detection_result_label = tk.Label(corner_detection_frame, text="Corners not detected.", anchor="w")
        self.corner_detection_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Wall Slicer Frame
        wall_slice_frame = tk.LabelFrame(right_column, text="Wall Slice Creation")
        wall_slice_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            wall_slice_frame.grid_columnconfigure(i, weight=1)

        tk.Label(wall_slice_frame, text="Slice Height").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.slice_height_entry = tk.Entry(
            wall_slice_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.slice_height_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(wall_slice_frame, text="Search Radius").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.search_radius_entry = tk.Entry(
            wall_slice_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.search_radius_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.wall_slice_button = tk.Button(
            wall_slice_frame,
            text="Create Wall Slice",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_wall_slice_thread
        )
        self.wall_slice_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.wall_slice_result_label = tk.Label(wall_slice_frame, text="Wall slice not created.", anchor="w")
        self.wall_slice_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Roof Slicer Frame
        roof_slice_frame = tk.LabelFrame(right_column, text="Roof Slicing")
        roof_slice_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            roof_slice_frame.grid_columnconfigure(i, weight=1)

        tk.Label(roof_slice_frame, text="Roof Layers").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.roof_layers_entry = tk.Entry(
            roof_slice_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.roof_layers_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_slice_frame, text="Layer Fatness").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.layer_fatness_entry = tk.Entry(
            roof_slice_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.layer_fatness_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_slice_frame, text="Search Radius").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.roof_search_radius_entry = tk.Entry(
            roof_slice_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_search_radius_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Roof extraction and slicing buttons
        self.roof_extraction_button = tk.Button(
            roof_slice_frame,
            text="Extract Roof Points",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_roof_extraction_thread
        )
        self.roof_extraction_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew")

        self.roof_slice_button = tk.Button(
            roof_slice_frame,
            text="Slice Roof",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_roof_slice_thread
        )
        self.roof_slice_button.grid(row=1, column=2, padx=5, pady=2, sticky="ew")

        self.roof_outline_button = tk.Button(
            roof_slice_frame,
            text="Find Roof Outline",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_roof_outline_thread
        )
        self.roof_outline_button.grid(row=2, column=2, padx=5, pady=2, sticky="ew")

        # Roof processing result labels
        self.roof_extraction_result_label = tk.Label(
            roof_slice_frame,
            text="Roof points not extracted.",
            anchor="w",
            font=("Arial", 8)
        )
        self.roof_extraction_result_label.grid(row=3, column=0, columnspan=3, padx=5, pady=2, sticky="ew")

        self.roof_slice_result_label = tk.Label(roof_slice_frame, text="Roof not sliced.", anchor="w", font=("Arial", 8))
        self.roof_slice_result_label.grid(row=4, column=0, columnspan=3, padx=5, pady=2, sticky="ew")

        self.roof_outline_result_label = tk.Label(roof_slice_frame, text="Roof outline not found.", anchor="w", font=("Arial", 8))
        self.roof_outline_result_label.grid(row=5, column=0, columnspan=3, padx=5, pady=2, sticky="ew")

        # Misc Frame (Final Actions)
        misc_frame = tk.LabelFrame(right_column, text="Final Actions")
        misc_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            misc_frame.grid_columnconfigure(i, weight=1)

        self.combine_results_button = tk.Button(
            misc_frame,
            text="Combine Results",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_combine_results_thread
        )
        self.combine_results_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.view_button = tk.Button(
            misc_frame,
            text="View Result",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.view_pointcloud
        )
        self.view_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.reset_button = tk.Button(misc_frame, text="Reset All", width=self.button_width, command=self.reset_application)
        self.reset_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.combine_results_result_label = tk.Label(misc_frame, text="Results not combined.", anchor="w")
        self.combine_results_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Exit Button (spans full width)
        exit_button = tk.Button(main_frame, text="Back", width=self.button_width, command=self.close_window)
        exit_button.pack(pady=10, fill=tk.X)

        self.root.bind("<Escape>", lambda event: exit_button.invoke())

    # Section enabling/disabling functions
    def disable_section(self, button, label_text):
        button.config(state=tk.DISABLED, text=label_text)

    def disable_all_sections(self):
        # Reset all buttons and labels
        self.preprocessing_button.config(state=tk.NORMAL, text="Start Preprocessing")
        self.preprocessing_result_label.config(text="Preprocessing not started.")

        self.heightmap_button.config(state=tk.DISABLED, text="Create Heightmap")
        self.heightmap_result_label.config(text="Heightmap not created.")
        self.grid_size_entry.config(state=tk.DISABLED)

        self.floor_detection_button.config(state=tk.DISABLED, text="Detect Floor Boundary")
        self.floor_detection_result_label.config(text="Floor boundary not detected.")
        self.alpha_value_entry.config(state=tk.DISABLED)
        self.triangle_size_entry.config(state=tk.DISABLED)

        self.corner_detection_button.config(state=tk.DISABLED, text="Detect Corners")
        self.corner_detection_result_label.config(text="Corners not detected.")
        self.distance_threshold_entry.config(state=tk.DISABLED)
        self.angle_threshold_entry.config(state=tk.DISABLED)
        self.merge_radius_entry.config(state=tk.DISABLED)

        self.wall_slice_button.config(state=tk.DISABLED, text="Create Wall Slice")
        self.wall_slice_result_label.config(text="Wall slice not created.")
        self.slice_height_entry.config(state=tk.DISABLED)
        self.search_radius_entry.config(state=tk.DISABLED)

        self.roof_extraction_button.config(state=tk.DISABLED, text="Extract Roof Points")
        self.roof_extraction_result_label.config(text="Roof points not extracted.")

        self.roof_slice_button.config(state=tk.DISABLED, text="Slice Roof")
        self.roof_slice_result_label.config(text="Roof not sliced.")
        self.roof_layers_entry.config(state=tk.DISABLED)
        self.layer_fatness_entry.config(state=tk.DISABLED)

        self.roof_outline_button.config(state=tk.DISABLED, text="Find Roof Outline")
        self.roof_outline_result_label.config(text="Roof outline not found.")
        self.roof_search_radius_entry.config(state=tk.DISABLED)

        self.combine_results_button.config(state=tk.DISABLED, text="Combine Results")
        self.combine_results_result_label.config(text="Results not combined.")

        self.view_button.config(state=tk.DISABLED)

    def enable_heightmap_section(self):
        self.heightmap_button.config(state=tk.NORMAL, text="Create Heightmap")
        self.grid_size_entry.config(state=tk.NORMAL)

    def enable_floor_detection_section(self):
        self.floor_detection_button.config(state=tk.NORMAL, text="Detect Floor Boundary")
        self.alpha_value_entry.config(state=tk.NORMAL)
        self.triangle_size_entry.config(state=tk.NORMAL)

    def enable_corner_detection_section(self):
        self.corner_detection_button.config(state=tk.NORMAL, text="Detect Corners")
        self.distance_threshold_entry.config(state=tk.NORMAL)
        self.angle_threshold_entry.config(state=tk.NORMAL)
        self.merge_radius_entry.config(state=tk.NORMAL)

    def enable_wall_slice_section(self):
        self.wall_slice_button.config(state=tk.NORMAL, text="Create Wall Slice")
        self.slice_height_entry.config(state=tk.NORMAL)
        self.search_radius_entry.config(state=tk.NORMAL)

    def enable_roof_extraction_section(self):
        self.roof_extraction_button.config(state=tk.NORMAL, text="Extract Roof Points")

    def enable_roof_slice_section(self):
        self.roof_slice_button.config(state=tk.NORMAL, text="Slice Roof")
        self.roof_layers_entry.config(state=tk.NORMAL)
        self.layer_fatness_entry.config(state=tk.NORMAL)

    def enable_roof_outline_section(self):
        self.roof_outline_button.config(state=tk.NORMAL, text="Find Roof Outline")
        self.roof_search_radius_entry.config(state=tk.NORMAL)

    def enable_combine_results_section(self):
        self.combine_results_button.config(state=tk.NORMAL, text="Combine Results")

    def enable_final_actions(self):
        self.view_button.config(state=tk.NORMAL)

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

        if os.path.exists(presets_file):
            config.read(presets_file)
            try:
                # Load default values
                self.voxel_size_entry.insert(0, config.get('Settings', 'voxel_size', fallback='0.05'))
                self.neighbour_amount_entry.insert(0, config.get('Settings', 'neighbour_amount', fallback='20'))
                self.std_ratio_entry.insert(0, config.get('Settings', 'std_ratio', fallback='2.0'))
                self.grid_size_entry.insert(0, config.get('Settings', 'grid_size', fallback='500'))
                self.alpha_value_entry.insert(0, config.get('Settings', 'alpha_value', fallback='11'))
                self.triangle_size_entry.insert(0, config.get('Settings', 'triangle_size', fallback='1e-10'))
                self.distance_threshold_entry.insert(0, config.get('Settings', 'distance_threshold', fallback='0.05'))
                self.angle_threshold_entry.insert(0, config.get('Settings', 'angle_threshold', fallback='45'))
                self.merge_radius_entry.insert(0, config.get('Settings', 'merge_radius', fallback='1'))
                self.slice_height_entry.insert(0, config.get('Settings', 'slice_height', fallback='1.5'))
                self.search_radius_entry.insert(0, config.get('Settings', 'search_radius', fallback='0.025'))
                self.roof_layers_entry.insert(0, config.get('Settings', 'roof_layers', fallback='5'))
                self.layer_fatness_entry.insert(0, config.get('Settings', 'layer_fatness', fallback='0.0075'))
                self.roof_search_radius_entry.insert(0, config.get('Settings', 'roof_search_radius', fallback='0.025'))
            except Exception as e:
                print(f"Error loading presets: {e}")
        else:
            # Set default values if no presets file exists
            self.voxel_size_entry.insert(0, '0.05')
            self.neighbour_amount_entry.insert(0, '20')
            self.std_ratio_entry.insert(0, '2.0')
            self.grid_size_entry.insert(0, '500')
            self.alpha_value_entry.insert(0, '11')
            self.triangle_size_entry.insert(0, '1e-10')
            self.distance_threshold_entry.insert(0, '0.05')
            self.angle_threshold_entry.insert(0, '45')
            self.merge_radius_entry.insert(0, '1')
            self.slice_height_entry.insert(0, '1.5')
            self.search_radius_entry.insert(0, '0.025')
            self.roof_layers_entry.insert(0, '5')
            self.layer_fatness_entry.insert(0, '0.0075')
            self.roof_search_radius_entry.insert(0, '0.025')

    def close_window(self):
        self.root.destroy()


def main():
    root = tk.Tk()
    app = App(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
