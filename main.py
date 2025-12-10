import configparser
import json
import locale
import os
import open3d as o3d
from random import randint as KernelMan
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox
from tqdm import tqdm

# Fix for PyInstaller windowed mode - redirect stdout/stderr if they're None
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from Source.fileHandler import (  # noqa: F401
    get_file_path,
    readout_LAS_file,
    get_save_file_path
)
from Source.floorplanFinder import find_boundary_from_floor, sort_points_in_hull
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
        self.timer_id = None
        self.check_id = None

        widget.bind("<Enter>", self.on_enter, add="+")
        widget.bind("<Leave>", self.on_leave, add="+")
        widget.bind("<Button>", self.on_leave, add="+")

        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def on_enter(self, event=None):
        if not self.text:
            return
        self.schedule_show()

    def on_leave(self, event=None):
        self.cancel_all()
        self.destroy_tooltip()

    def schedule_show(self):
        self.cancel_timer()
        try:
            self.timer_id = self.widget.after(1500, self.show_tooltip)
        except Exception:
            pass

    def show_tooltip(self):
        self.timer_id = None

        if self.tipwindow:
            return

        try:
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 20

            self.tipwindow = tk.Toplevel(self.widget)
            self.tipwindow.wm_overrideredirect(True)
            self.tipwindow.wm_geometry(f"+{x}+{y}")

            label = tk.Label(
                self.tipwindow,
                text=self.text,
                justify='left',
                background="#ffffe0",
                relief='solid',
                borderwidth=1,
                font=("tahoma", "8", "normal")
            )
            label.pack(ipadx=1)

            self.check_position()
        except Exception:
            self.destroy_tooltip()

    def check_position(self):
        if not self.tipwindow:
            self.check_id = None
            return

        try:
            mouse_x = self.widget.winfo_pointerx()
            mouse_y = self.widget.winfo_pointery()
            widget_x = self.widget.winfo_rootx()
            widget_y = self.widget.winfo_rooty()
            widget_w = self.widget.winfo_width()
            widget_h = self.widget.winfo_height()

            x_in = widget_x <= mouse_x <= widget_x + widget_w
            y_in = widget_y <= mouse_y <= widget_y + widget_h

            if not (x_in and y_in):
                self.destroy_tooltip()
                return

            self.check_id = self.widget.after(100, self.check_position)
        except Exception:
            self.destroy_tooltip()

    def cancel_timer(self):
        if self.timer_id:
            try:
                self.widget.after_cancel(self.timer_id)
            except Exception:
                pass
            self.timer_id = None

    def cancel_check(self):
        if self.check_id:
            try:
                self.widget.after_cancel(self.check_id)
            except Exception:
                pass
            self.check_id = None

    def cancel_all(self):
        self.cancel_timer()
        self.cancel_check()

    def destroy_tooltip(self):
        self.cancel_check()

        if self.tipwindow:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None


class App:
    def __init__(self, root, point_cloud_data=None, point_cloud_path=None):
        self.root = root
        self.root.title("Pygmalion CityJSON Generator")
        self.root.resizable(False, False)
        self.root.iconbitmap("Source\\support_files\\logo.ico")
        self.root.geometry("+20+20")  # Open window in the top-left corner

        # Store the point cloud data and path
        self.point_cloud_data = point_cloud_data
        self.point_cloud_path = point_cloud_path

        # Tooltip tracking
        self.tooltips = []

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

        self.lineset_preview = None
        self.mesh_preview = None

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
        self.load_presets()

        # If point cloud data is provided, load it
        if self.point_cloud_data is not None and self.point_cloud_path is not None:
            self.load_point_cloud_data()

        # Schedule periodic internal validation
        self._schedule_integrity_check()

        # Schedule periodic tooltip reset
        self._schedule_tooltip_reset()

    def _schedule_tooltip_reset(self):
        """Reset all tooltips every 5 seconds to prevent stuck states"""
        self.reset_all_tooltips()
        self.root.after(5000, self._schedule_tooltip_reset)

    def reset_all_tooltips(self):
        """Reset all tooltip instances to their initial state"""
        for tooltip in self.tooltips:
            try:
                tooltip.reset()
            except Exception:
                pass

    def add_tooltip(self, widget, text):
        """Create a tooltip and track it"""
        tooltip = Tooltip(widget, text)
        self.tooltips.append(tooltip)
        return tooltip

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

    def validate_empty_field(self, entry_widget):
        """
        Validates that an entry field is not empty.
        Automatically extracts field name from widget attribute name.
        """
        if entry_widget.get() == "":
            # Extract the field name from the widget's attribute name
            field_name = None
            for attr_name in dir(self):
                if getattr(self, attr_name) is entry_widget:
                    # Convert from snake_case to readable format, but only capitalize first letter
                    field_name = attr_name.replace('_entry', '').replace('_', ' ').capitalize()
                    break

            if field_name is None:
                field_name = "Field"

            raise ValueError(f"Voor een waarde in voor veld: {field_name}.")

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

    def select_file(self):
        """Select a point cloud file and load it"""
        try:
            # Update button and label text
            self.file_select_button.config(text="Bestand laden...")
            self.file_label.config(text="Bestand wordt geladen...")
            self.point_amount_label.config(text="Punten worden geteld...")

            file_path = get_file_path("Puntenwolk bestanden", ["*.las", "*.laz"], False)

            if file_path:
                self.point_cloud_path = file_path

                # Update the file label
                self.file_label.config(text=f"Geselecteerd bestand: {os.path.basename(file_path)}")

                # Load the point cloud data
                self.point_cloud_data = readout_LAS_file(file_path, False)

                self.file_select_button.config(text="Bestand wijzigen")

                # Enable point density section
                self.enable_point_density_section()

                # Enable view button
                self.enable_view_pointcloud(self.point_cloud_data)

                # Update the file label with file name and point count
                self.file_label.config(text=f"Geselecteerd bestand: {os.path.basename(file_path)}")
                self.point_amount_label.config(text=f"Punten: {len(self.point_cloud_data.points):n}")

            else:
                # User cancelled file selection
                self.show_message("Info", "Bestand selectie geannuleerd.", "info")
                self.file_select_button.config(text="Bestand selecteren")
                self.file_label.config(text="Geen bestand geselecteerd.")
                self.point_amount_label.config(text="")

        except Exception as e:
            # Bring back the file select button text
            self.file_select_button.config(text="Bestand selecteren")
            self.point_amount_label.config(text="")

            self.show_message("Foutmelding", f"Fout bij laden van puntenwolkbestand: {str(e)}", "error")

            # Update the file label with an error message
            self.file_label.config(
                text=f"Fout bij laden van puntenwolkbestand: {os.path.basename(file_path)}",
                fg="red"
            )

    def load_point_cloud_data(self):
        """Load point cloud data when provided during initialization"""
        if self.point_cloud_path and os.path.exists(self.point_cloud_path):
            self.file_label.config(text=f"Geselecteerd bestand:{os.path.basename(self.point_cloud_path)}")
            self.file_select_button.config(text="Bestand wijzigen")
            self.update_view_pointcloud(self.point_cloud_data)
            self.enable_point_density_section()

    # Threading functions for each step
    def start_alter_point_density_thread(self):
        if not self.point_cloud_data:
            self.point_density_result_label.config(text="Puntdichtheid aanpassen, even geduld...")
            self.show_message("Waarschuwing", "Selecteer eerst een puntenwolkbestand.", "warning")
            return
        self.root.config(cursor="watch")
        self.disable_section(self.point_density_button, "Puntdichtheid aanpassen...")
        threading.Thread(target=self.alter_point_density_step).start()

    def start_preprocessing_thread(self):
        if not self.resized_point_cloud_data:
            self.preprocessing_result_label.config(text="Voorbewerking, even geduld...")
            self.show_message("Waarschuwing", "Voltooi eerst de stap puntdichtheid.", "warning")
            return
        self.root.config(cursor="watch")
        self.disable_section(self.preprocessing_button, "Voorbewerking...")
        threading.Thread(target=self.preprocessing_step).start()

    def start_heightmap_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.heightmap_button, "Hoogtekaart maken...")
        self.heightmap_result_label.config(text="Hoogtekaart maken, even geduld...")
        threading.Thread(target=self.heightmap_step).start()

    def start_floor_detection_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.floor_detection_button, "Vloer detecteren...")
        self.floor_detection_result_label.config(text="Vloer detecteren, even geduld...")
        threading.Thread(target=self.floor_detection_step).start()

    def start_floor_2_lineset_2_cityjson_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.floor_to_cityjson_button, "Vloer naar 2D CityJSON")
        self.floor_detection_result_label.config(text="Even geduld...")
        threading.Thread(target=self.floor_2_lineset_2_cityjson_step).start()

    def start_roof_extraction_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.roof_extraction_button, "Dak extractie...")
        self.roof_extraction_result_label.config(text="Dak extractie, even geduld...")
        threading.Thread(target=self.roof_extraction_step).start()

    def start_roof_division_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.roof_division_button, "Dak verdelen...")
        self.roof_division_result_label.config(text="Dak verdelen, even geduld...")
        threading.Thread(target=self.roof_division_step).start()

    def start_wall_extraction_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.wall_extraction_button, "Muren extraheren...")
        self.wall_extraction_result_label.config(text="Muren extraheren, even geduld...")
        threading.Thread(target=self.wall_extraction_step).start()

    def start_wall_division_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.wall_division_button, "Muren verdelen...")
        self.wall_division_result_label.config(text="Muren verdelen, even geduld...")
        threading.Thread(target=self.wall_division_step).start()

    def start_pcd_to_lineset_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.pcd_to_lineset_button, "Converteren naar\nLineset...")
        self.pcd_to_lineset_result_label.config(text="Converteren naar Lineset, even geduld...")
        threading.Thread(target=self.pcd_to_lineset_step).start()

    def start_lineset_to_mesh_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.lineset_to_mesh_button, "Converteren naar Mesh...")
        self.lineset_to_mesh_result_label.config(text="Converteren naar Mesh, even geduld...")
        threading.Thread(target=self.lineset_to_mesh_step).start()

    def start_repair_mesh_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.repair_mesh_button, "Repareren Mesh...")
        self.repair_mesh_result_label.config(text="Repareren Mesh, even geduld...")
        threading.Thread(target=self.repair_mesh_step).start()

    def start_cityjson_conversion_thread(self):
        self.root.config(cursor="watch")
        self.disable_section(self.cityjson_conversion_button, "Converteren naar CityJSON...")
        self.cityjson_conversion_result_label.config(text="Converteren naar CityJSON, even geduld...")
        threading.Thread(target=self.cityjson_conversion_step).start()

    # Processing steps
    def alter_point_density_step(self):
        # This piece of code will be repeated so that when the steps are ran again and the lineset/mesh preview is active,
        # it will switch back to point cloud view and not show the lineset/mesh preview.
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            self.validate_empty_field(self.points_per_cm_entry)

            resized_pcd = alter_point_density(
                self.point_cloud_data,
                points_per_cm=float(self.points_per_cm_entry.get())
            )

            self.point_density_result_label.config(
                text=f"Puntdichtheid aangepast van {len(self.point_cloud_data.points):n} → {len(resized_pcd.points):n} punten."
            )

            self.resized_point_cloud_data = resized_pcd
            self.point_density_button.config(state=tk.NORMAL, text="Puntdichtheid aanpassen")
            self.update_view_pointcloud(resized_pcd)
            self.enable_preprocessing_section()
            self.root.config(cursor="")
        except Exception as e:
            self.point_density_result_label.config(text=f"Fout: {str(e)}")
            self.point_density_button.config(state=tk.NORMAL, text="Puntdichtheid aanpassen")
            self.root.config(cursor="")

    def preprocessing_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            pcd = self.resized_point_cloud_data

            self.validate_empty_field(self.neighbour_amount_entry)
            self.validate_empty_field(self.std_ratio_entry)

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
            self.preprocessing_result_label.config(text=f"{amount_removed} punten verwijderd, {len(pcd.points)} punten over.")
            self.preprocessing_button.config(state=tk.NORMAL, text="Start voorbewerking")
            self.update_view_pointcloud(pcd)
            self.enable_heightmap_section()
            self.root.config(cursor="")
        except Exception as e:
            self.preprocessing_result_label.config(text=f"Fout: {str(e)}")
            self.preprocessing_button.config(state=tk.NORMAL, text="Start voorbewerking")
            self.root.config(cursor="")

    def heightmap_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            self.new_pcd_tuple = transform_pointcloud_to_height_map(
                self.processed_pcd,
                visualize_map=self.visualize_heightmap_var.get(),
                visualize_map_np=False,
                debugging_logs=False
            )
            self.heightmap_result_label.config(text="Hoogtekaart succesvol aangemaakt.")
            self.heightmap_button.config(state=tk.NORMAL, text="Hoogtekaart maken")
            self.update_view_pointcloud(self.new_pcd_tuple[0])
            self.enable_floor_detection_section()
            self.root.config(cursor="")
        except Exception as e:
            self.heightmap_result_label.config(text=f"Fout: {str(e)}")
            self.heightmap_button.config(state=tk.NORMAL, text="Hoogtekaart maken")
            self.root.config(cursor="")

    def floor_detection_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        # Check if user filled in alpha_value and triangle_size
        self.validate_empty_field(self.floor_alpha_value_entry)
        self.validate_empty_field(self.floor_triangle_size_entry)
        self.validate_empty_field(self.corner_distance_threshold_entry)

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
                text=f"Vloergrens gedetecteerd. {len(self.floor_lines)} grenspunten, {len(self.floor_corners)} hoeken."
            )
            self.floor_detection_button.config(state=tk.NORMAL, text="Detecteer vloergrens")
            self.update_view_pointcloud(create_point_cloud(self.floor_corners, color=[1, 0, 0]))
            self.enable_floor_to_cityjson_section()
            self.enable_roof_extraction_section()
            self.root.config(cursor="")
        except Exception as e:
            self.floor_detection_result_label.config(text=f"Fout: {str(e)}")
            self.floor_detection_button.config(state=tk.NORMAL, text="Detecteer vloergrens")
            self.root.config(cursor="")

    def floor_2_lineset_2_cityjson_step(self):
        # In this function we convert the floor to a lineset, then to a mesh and then to a cityjson object. All within one step.
        # There won't be a preview for this step, as it's just a combination of already existing steps.
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            # Check if we received a config value for max_line_length
            if self.validate_empty_field(self.max_line_length_entry):
                max_line_length = float(self.max_line_length_entry.get())
                floor_lineset = contour_to_lineset(self.floor_corners, max_line_length=max_line_length)
            else:
                floor_lineset = contour_to_lineset(self.floor_corners)
            # No need for x-y tolerance or max line length here, as it's just the floor (it's flat anyway)
            floor_mesh = lineset_to_trianglemesh(floor_lineset, self.floor_corners)
            cityjson_data = o3d_to_cityjson(
                floor_mesh,
                cityobject_id="Gebouw_Vloer_1",
                obj_type="Building",
                lod="1.0"
            )

            self.floor_detection_result_label.config(text="Vloer succesvol geconverteerd naar CityJSON.")
            self.floor_to_cityjson_button.config(state=tk.NORMAL, text="Vloer naar 2D CityJSON")

            # Store the cityjson data for saving later
            self.cityjson_data = cityjson_data

            self.root.config(cursor="")

            # Enable save button
            self.save_cityjson_button.config(state=tk.NORMAL)
        except Exception as e:
            self.floor_detection_result_label.config(text=f"Fout: {str(e)}")
            self.floor_to_cityjson_button.config(state=tk.NORMAL, text="Vloer naar 2D CityJSON")
            self.root.config(cursor="")

    def roof_extraction_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            self.validate_empty_field(self.slice_height_entry)

            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])

            # Use define_min_height_roof which returns roof_pcd and other_wall_pcd
            self.roof_pcd, self.temp_wall_pcd = define_min_height_roof(
                self.new_pcd_tuple[1],
                floor_corners_pcd,
                height=float(self.slice_height_entry.get())
            )

            self.roof_extraction_result_label.config(
                text=f"Dak geëxtraheerd: {len(self.roof_pcd.points)} dakpunten, {len(self.temp_wall_pcd.points)} muurpunten."
            )
            self.roof_extraction_button.config(state=tk.NORMAL, text="Extraheer dakpunten")
            self.update_view_pointcloud(self.roof_pcd)
            self.enable_roof_division_section()
            self.root.config(cursor="")
        except Exception as e:
            self.roof_extraction_result_label.config(text=f"Fout: {str(e)}")
            self.roof_extraction_button.config(state=tk.NORMAL, text="Extraheer dakpunten")
            self.root.config(cursor="")

    def roof_division_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            # Set default values if not provided
            self.validate_empty_field(self.roof_layers_entry)
            self.validate_empty_field(self.roof_layer_fatness_entry)
            self.validate_empty_field(self.roof_voxel_size_entry)
            self.validate_empty_field(self.roof_merge_radius_entry)
            self.validate_empty_field(self.roof_angle_threshold_entry)

            self.roof_layer_list = slice_roof_up(
                self.roof_pcd,
                slices_amount=int(self.roof_layers_entry.get()),
                slab_fatness=float(self.roof_layer_fatness_entry.get()),
                voxel_size=float(self.roof_voxel_size_entry.get())
            )

            self.roof_division_result_label.config(
                text=f"Dak verdeeld in {len(self.roof_layer_list)} lagen."
            )
            self.roof_division_button.config(state=tk.NORMAL, text="Verdeel dak")
            self.enable_wall_extraction_section()
            self.root.config(cursor="")
        except Exception as e:
            self.roof_division_result_label.config(text=f"Fout: {str(e)}")
            self.roof_division_button.config(state=tk.NORMAL, text="Verdeel dak")
            self.root.config(cursor="")

    def wall_extraction_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        try:
            self.validate_empty_field(self.wall_search_radius_entry)

            # Merge new_pcd_tuple[2] with temp_wall_pcd
            temp_wall_pcd_merged = merge_pcds([self.new_pcd_tuple[2], self.temp_wall_pcd])

            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])
            self.wall_pcd = extract_wall_points(
                temp_wall_pcd_merged,
                floor_corners_pcd,
                search_radius=float(self.wall_search_radius_entry.get())
            )

            self.wall_extraction_result_label.config(
                text=f"Muren geëxtraheerd: {len(self.wall_pcd.points)} muurpunten."
            )
            self.wall_extraction_button.config(state=tk.NORMAL, text="Extraheer muren")
            self.update_view_pointcloud(self.wall_pcd)
            self.enable_wall_division_section()
            self.root.config(cursor="")
        except Exception as e:
            self.wall_extraction_result_label.config(text=f"Fout: {str(e)}")
            self.wall_extraction_button.config(state=tk.NORMAL, text="Extraheer muren")
            self.root.config(cursor="")

    def wall_division_step(self):
        try:
            self.validate_empty_field(self.wall_layer_amount_entry)

            self.wall_layer_list = divide_wall_into_layers(
                self.wall_pcd,
                layer_amount=int(self.wall_layer_amount_entry.get())
            )

            self.wall_division_result_label.config(
                text=f"Muren verdeeld in {len(self.wall_layer_list)} lagen."
            )
            self.wall_division_button.config(state=tk.NORMAL, text="Verdeel muren")
            self.enable_pcd_to_lineset_section()
            self.root.config(cursor="")
        except Exception as e:
            self.wall_division_result_label.config(text=f"Fout: {str(e)}")
            self.wall_division_button.config(state=tk.NORMAL, text="Verdeel muren")
            self.root.config(cursor="")

    def pcd_to_lineset_step(self):
        self.mesh_preview = None

        try:
            self.validate_empty_field(self.xy_tolerance_entry)
            self.validate_empty_field(self.max_line_length_entry)

            self.roof_wall_lineset = o3d.geometry.LineSet()

            # Connect wall layers
            for i in tqdm(range(len(self.wall_layer_list)), desc="Bereken muur lagen", unit="laag"):
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
                text="Linesets succesvol aangemaakt."
            )

            self.lineset_preview = True

            self.pcd_to_lineset_button.config(state=tk.NORMAL, text="Converteer naar Lineset")
            self.enable_lineset_to_mesh_section()
            self.root.config(cursor="")
        except Exception as e:
            self.pcd_to_lineset_result_label.config(text=f"Fout: {str(e)}")
            self.pcd_to_lineset_button.config(state=tk.NORMAL, text="Converteer naar Lineset")
            self.root.config(cursor="")

    def lineset_to_mesh_step(self):
        try:
            self.floor_mesh = lineset_to_trianglemesh(self.floor_lineset, self.floor_corners)
            self.roof_wall_mesh = lineset_to_trianglemesh(self.total_lineset, self.floor_corners)

            self.lineset_preview = False
            self.mesh_preview = combine_meshes([self.floor_mesh, self.roof_wall_mesh])

            self.lineset_to_mesh_result_label.config(text="Meshes succesvol aangemaakt.")
            self.lineset_to_mesh_button.config(state=tk.NORMAL, text="Converteer naar Mesh")

            self.enable_repair_mesh_section()
            self.root.config(cursor="")
        except Exception as e:
            self.lineset_to_mesh_result_label.config(text=f"Fout: {str(e)}")
            self.lineset_to_mesh_button.config(state=tk.NORMAL, text="Converteer naar Mesh")
            self.root.config(cursor="")

    def repair_mesh_step(self):
        try:
            self.repaired_mesh = repair_mesh([self.roof_wall_mesh, self.floor_mesh])

            self.mesh_preview = self.repaired_mesh

            self.repair_mesh_result_label.config(text="Mesh succesvol hersteld.")
            self.repair_mesh_button.config(state=tk.NORMAL, text="Repareer Mesh")
            self.enable_cityjson_conversion_section()
            self.root.config(cursor="")
        except Exception as e:
            self.repair_mesh_result_label.config(text=f"Fout: {str(e)}")
            self.repair_mesh_button.config(state=tk.NORMAL, text="Repareer Mesh")
            self.root.config(cursor="")

    def cityjson_conversion_step(self):
        try:
            self.cityjson_data = o3d_to_cityjson(
                self.repaired_mesh,
                cityobject_id="Gebouw_1",
                obj_type="Building",
                lod="1.0"
            )

            self.cityjson_conversion_result_label.config(text="Succesvol geconverteerd naar CityJSON.")
            self.cityjson_conversion_button.config(state=tk.NORMAL, text="Converteer naar CityJSON")
            self.save_cityjson_button.config(state=tk.NORMAL)
            self.root.config(cursor="")
        except Exception as e:
            self.cityjson_conversion_result_label.config(text=f"Fout: {str(e)}")
            self.cityjson_conversion_button.config(state=tk.NORMAL, text="Converteer naar CityJSON")
            self.root.config(cursor="")

    def save_cityjson_file_step(self):
        try:
            if self.cityjson_data is None:
                self.show_message("Waarschuwing", "Geen CityJSON-gegevens om op te slaan. Voltooi eerst de conversiestap.", "warning")  # noqa: E501
                return

            save_path = get_save_file_path("CityJSON-bestanden", ["*.json"], default_name="building_cityjson.json")

            # Check if path ends with .json, if not, add it
            if save_path and not save_path.lower().endswith(".json"):
                save_path += ".json"

            if save_path:
                with open(save_path, "w") as f:
                    json.dump(self.cityjson_data, f, indent=2)

                self.show_message("Info", f"CityJSON-gegevens succesvol opgeslagen in {save_path}.", "info")

        except Exception as e:
            self.show_message("Fout", f"Opslaan van CityJSON-bestand mislukt: {str(e)}", "error")

    def view_pointcloud(self, pointcloud):
        # NOTE: Jumping back and forth between steps of LineSet and/or Mesh can cause an infinite loop.
        # It's caused by a what-looks-like Open3d bug where the draw function keeps calling itself when
        # while it's telling itself that the GLFW library is not initialized.

        # NOTE: This doesn't work when exporting to an executable with PyInstaller due to Open3D limitations.

        # This will be called when the pointcloud is changed to a lineset
        if self.lineset_preview is True and self.total_lineset is not None:
            o3d.visualization.draw([self.total_lineset])

        # This will be called when the Lineset is changed to a mesh
        elif self.mesh_preview is not None:
            o3d.visualization.draw([self.mesh_preview])

        elif pointcloud is not None:
            opce(pointcloud, False)
        else:
            self.show_message("Waarschuwing", "Geen puntenwolk om te bekijken.", "warning")

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

        self.lineset_preview = None
        self.mesh_preview = None

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

        # Reset all sections
        self.disable_all_sections()

        self.load_presets()

        self.show_message("Info", "Applicatie succesvol gereset.")

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
        file_frame = tk.LabelFrame(left_column, text="Bestand selecteren")
        file_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            file_frame.grid_columnconfigure(i, weight=1, uniform="col_with_button")

        self.point_amount_label = tk.Label(file_frame, text="", anchor="w")
        self.point_amount_label.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.file_label = tk.Label(file_frame, text="Geen bestand geselecteerd", anchor="w")
        self.file_label.grid(row=1, column=0, padx=5, pady=5, sticky="ew", columnspan=2)

        self.file_select_button = tk.Button(
            file_frame,
            text="Selecteer puntenwolkbestand",
            command=self.select_file,
            anchor="center",
            justify="right",
            width=30
        )
        self.file_select_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew", rowspan=2)
        Tooltip(self.file_select_button, "Selecteer een .las of .laz puntenwolkbestand om te beginnen.")

        # Point Density Frame
        point_density_frame = tk.LabelFrame(left_column, text="Puntdichtheid")
        point_density_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            point_density_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(point_density_frame, text="Punten per cm²", anchor="w").grid(
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
            text="Pas puntdichtheid aan",
            command=self.start_alter_point_density_thread,
            state=tk.DISABLED,
            width=30
        )
        self.point_density_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew", rowspan=2)

        self.point_density_result_label = tk.Label(point_density_frame, text="", anchor="w")
        self.point_density_result_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Noise Removal Frame
        noise_removal_frame = tk.LabelFrame(left_column, text="Ruis verwijderen")
        noise_removal_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            noise_removal_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(noise_removal_frame, text="Aantal buren", anchor="w").grid(
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

        tk.Label(noise_removal_frame, text="Std ratio", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
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
            text="Toon verwijderde punten",
            variable=self.show_removed_points_var,
            state=tk.DISABLED,
            anchor="w"
        )
        self.show_removed_points_checkbox.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.preprocessing_button = tk.Button(
            noise_removal_frame,
            text="Start voorbewerking",
            command=self.start_preprocessing_thread,
            state=tk.DISABLED,
            width=30
        )
        self.preprocessing_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.preprocessing_result_label = tk.Label(noise_removal_frame, text="", anchor="w")
        self.preprocessing_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Heightmap Frame
        heightmap_frame = tk.LabelFrame(left_column, text="Hoogtekaart genereren")
        heightmap_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            heightmap_frame.grid_columnconfigure(i, weight=1, uniform="col")

        # Add visualization checkbox for heightmap
        self.visualize_heightmap_var = tk.BooleanVar()
        self.visualize_heightmap_checkbox = tk.Checkbutton(
            heightmap_frame,
            text="Visualiseer kaart",
            variable=self.visualize_heightmap_var,
            state=tk.DISABLED,
            anchor="w"
        )
        self.visualize_heightmap_checkbox.grid(
            row=0,
            column=0,
            padx=5,
            pady=5,
            sticky="ew",
            columnspan=2
        )

        self.heightmap_button = tk.Button(
            heightmap_frame,
            text="Maak hoogtekaart",
            state=tk.DISABLED,
            command=self.start_heightmap_thread,
            width=30
        )
        self.heightmap_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew", rowspan=2)

        self.heightmap_result_label = tk.Label(heightmap_frame, text="Hoogtekaart niet gemaakt.", anchor="w")
        self.heightmap_result_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Floor Detection Frame
        floor_detection_frame = tk.LabelFrame(left_column, text="Vloergrens detectie")
        floor_detection_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            floor_detection_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(floor_detection_frame, text="Alpha waarde", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.floor_alpha_value_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.floor_alpha_value_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(floor_detection_frame, text="Driehoekgrootte", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.floor_triangle_size_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.floor_triangle_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(floor_detection_frame, text="Afstandsdrempel", anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.corner_distance_threshold_entry = tk.Entry(
            floor_detection_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.corner_distance_threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        self.floor_detection_button = tk.Button(
            floor_detection_frame,
            text="Detecteer vloergrens",
            state=tk.DISABLED,
            command=self.start_floor_detection_thread,
            width=30
        )
        self.floor_detection_button.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky="nsew")

        self.floor_to_cityjson_button = tk.Button(
            floor_detection_frame,
            text="Vloer naar 2D CityJSON",
            state=tk.DISABLED,
            command=self.start_floor_2_lineset_2_cityjson_thread,
            width=30
        )
        self.floor_to_cityjson_button.grid(row=3, column=2, padx=5, pady=5, sticky="nsew")

        self.floor_detection_result_label = tk.Label(floor_detection_frame, text="Vloergrens niet gedetecteerd.", anchor="w")
        self.floor_detection_result_label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Roof Extraction Frame
        roof_extraction_frame = tk.LabelFrame(left_column, text="Dakextractie")
        roof_extraction_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            roof_extraction_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(roof_extraction_frame, text="Snijlaaghoogte", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.slice_height_entry = tk.Entry(
            roof_extraction_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.slice_height_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.roof_extraction_button = tk.Button(
            roof_extraction_frame,
            text="Extraheer dakpunten",
            state=tk.DISABLED,
            command=self.start_roof_extraction_thread,
            width=30
        )
        self.roof_extraction_button.grid(row=0, column=2, padx=5, pady=5, sticky="nsew", rowspan=2)

        self.roof_extraction_result_label = tk.Label(roof_extraction_frame, text="Dak niet geëxtraheerd.", anchor="w")
        self.roof_extraction_result_label.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # === RIGHT COLUMN CONTENT ===

        # Roof Division Frame
        roof_division_frame = tk.LabelFrame(right_column, text="Dakverdeling")
        roof_division_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            roof_division_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(roof_division_frame, text="Daklagen", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.roof_layers_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.roof_layers_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Laagdikte", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.roof_layer_fatness_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_layer_fatness_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Voxelgrootte", anchor="w").grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.roof_voxel_size_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_voxel_size_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(roof_division_frame, text="Hoekdrempel", anchor="w").grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        self.roof_angle_threshold_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_angle_threshold_entry.grid(row=1, column=2, padx=5, pady=5, sticky="ew")

        # tk.Label(roof_division_frame, text="Puntvenster", anchor="w").grid(row=2, column=2, padx=5, pady=5, sticky="ew")
        # self.roof_point_window_amount_entry = tk.Entry(
        #     roof_division_frame,
        #     validate="key",
        #     validatecommand=(self.validate_int, '%P'),
        #     state=tk.DISABLED
        # )
        # # Note: No grid placement for this entry as it's not used in slice_roof_up

        tk.Label(roof_division_frame, text="Koppelradius", anchor="w").grid(row=3, column=0, padx=5, pady=5, sticky="ew")
        self.roof_merge_radius_entry = tk.Entry(
            roof_division_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.roof_merge_radius_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

        self.roof_division_button = tk.Button(
            roof_division_frame,
            text="Verdeel dak",
            state=tk.DISABLED,
            command=self.start_roof_division_thread
        )
        self.roof_division_button.grid(row=3, column=2, padx=5, pady=5, sticky="ew")

        self.roof_division_result_label = tk.Label(roof_division_frame, text="Dak niet verdeeld.", anchor="w")
        self.roof_division_result_label.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Wall Extraction Frame
        wall_extraction_frame = tk.LabelFrame(right_column, text="Wandextractie")
        wall_extraction_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            wall_extraction_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(wall_extraction_frame, text="Zoekradius", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.wall_search_radius_entry = tk.Entry(
            wall_extraction_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.wall_search_radius_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.wall_extraction_button = tk.Button(
            wall_extraction_frame,
            text="Extraheer muren",
            state=tk.DISABLED,
            command=self.start_wall_extraction_thread
        )
        self.wall_extraction_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.wall_extraction_result_label = tk.Label(wall_extraction_frame, text="Muren niet geëxtraheerd.", anchor="w")
        self.wall_extraction_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Wall Division Frame
        wall_division_frame = tk.LabelFrame(right_column, text="Wandverdeling")
        wall_division_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            wall_division_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(wall_division_frame, text="Aantal lagen", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.wall_layer_amount_entry = tk.Entry(
            wall_division_frame,
            validate="key",
            validatecommand=(self.validate_int, '%P'),
            state=tk.DISABLED
        )
        self.wall_layer_amount_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.wall_division_button = tk.Button(
            wall_division_frame,
            text="Verdeel muren",
            state=tk.DISABLED,
            command=self.start_wall_division_thread
        )
        self.wall_division_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        self.wall_division_result_label = tk.Label(wall_division_frame, text="Muren niet verdeeld.", anchor="w")
        self.wall_division_result_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # PCD to Lineset Frame
        pcd_to_lineset_frame = tk.LabelFrame(right_column, text="Puntenwolk naar Lineset")
        pcd_to_lineset_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            pcd_to_lineset_frame.grid_columnconfigure(i, weight=1, uniform="col")

        tk.Label(pcd_to_lineset_frame, text="XY tolerantie", anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.xy_tolerance_entry = tk.Entry(
            pcd_to_lineset_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.xy_tolerance_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(pcd_to_lineset_frame, text="Max. lijnlengte", anchor="w").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.max_line_length_entry = tk.Entry(
            pcd_to_lineset_frame,
            validate="key",
            validatecommand=(self.validate_flt, '%P'),
            state=tk.DISABLED
        )
        self.max_line_length_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        self.pcd_to_lineset_button = tk.Button(
            pcd_to_lineset_frame,
            text="Converteer naar\nLineset",
            state=tk.DISABLED,
            command=self.start_pcd_to_lineset_thread
        )
        self.pcd_to_lineset_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.pcd_to_lineset_result_label = tk.Label(pcd_to_lineset_frame, text="Lineset niet gemaakt.", anchor="w")
        self.pcd_to_lineset_result_label.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="ew")

        # Lineset to Mesh Frame
        lineset_to_mesh_frame = tk.LabelFrame(right_column, text="Lineset naar Mesh")
        lineset_to_mesh_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            lineset_to_mesh_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.lineset_to_mesh_button = tk.Button(
            lineset_to_mesh_frame,
            text="Converteer naar Mesh",
            state=tk.DISABLED,
            command=self.start_lineset_to_mesh_thread
        )
        self.lineset_to_mesh_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.lineset_to_mesh_result_label = tk.Label(lineset_to_mesh_frame, text="Mesh niet gemaakt.", anchor="w")
        self.lineset_to_mesh_result_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Repair Mesh Frame
        repair_mesh_frame = tk.LabelFrame(right_column, text="Mesh reparatie")
        repair_mesh_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            repair_mesh_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.repair_mesh_button = tk.Button(
            repair_mesh_frame,
            text="Repareer Mesh",
            state=tk.DISABLED,
            command=self.start_repair_mesh_thread
        )
        self.repair_mesh_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.repair_mesh_result_label = tk.Label(repair_mesh_frame, text="Mesh niet gerepareerd.", anchor="w")
        self.repair_mesh_result_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # CityJSON Conversion Frame
        cityjson_conversion_frame = tk.LabelFrame(right_column, text="CityJSON conversie")
        cityjson_conversion_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            cityjson_conversion_frame.grid_columnconfigure(i, weight=1, uniform="col")

        self.cityjson_conversion_button = tk.Button(
            cityjson_conversion_frame,
            text="Converteer naar CityJSON",
            state=tk.DISABLED,
            command=self.start_cityjson_conversion_thread
        )
        self.cityjson_conversion_button.grid(row=0, column=2, rowspan=2, padx=5, pady=5, sticky="nsew")

        self.cityjson_conversion_result_label = tk.Label(
            cityjson_conversion_frame,
            text="Niet geconverteerd naar CityJSON.",
            anchor="w"
        )
        self.cityjson_conversion_result_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Misc Frame (Final Actions)
        misc_frame = tk.LabelFrame(main_frame, text="Eindacties")
        misc_frame.pack(fill="x", pady=5, padx=10)
        for i in range(3):
            misc_frame.grid_columnconfigure(i, weight=1, uniform="col")

        # View Button
        self.view_button = tk.Button(
            misc_frame,
            text="Bekijk resultaat",
            state=tk.DISABLED,
            command=lambda: None  # Default disabled command
        )
        self.view_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        Tooltip(self.view_button, "Bekijk het huidige resultaat in een nieuw venster.")

        # Save CityJSON Button
        self.save_cityjson_button = tk.Button(
            misc_frame,
            text="Sla CityJSON op",
            state=tk.DISABLED,
            command=self.save_cityjson_file_step
        )
        self.save_cityjson_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        Tooltip(self.save_cityjson_button, "Sla het gegenereerde CityJSON-bestand op.")

        # Reset Button
        self.reset_button = tk.Button(misc_frame, text="Reset alles", command=self.reset_application)
        self.reset_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        Tooltip(self.reset_button, "Reset de applicatie naar de begintoestand.")

        # Exit Button
        exit_button = tk.Button(main_frame, text="Terug", command=self.on_close)
        exit_button.pack(pady=10, fill=tk.X)
        Tooltip(exit_button, "Sluit de applicatie.")

        self.root.bind("<Escape>", lambda event: self.on_close())  # Bind Escape key to close the window

        # Make all input fields on left column 20 wide
        for child in left_column.winfo_children():
            if isinstance(child, tk.LabelFrame):
                for subchild in child.winfo_children():
                    if isinstance(subchild, tk.Entry):
                        subchild.config(width=20)

    # Section enabling/disabling functions
    def disable_section(self, button, label_text):
        button.config(state=tk.DISABLED, text=label_text)
        Tooltip(button, "")

    def disable_all_sections(self):
        "Disable all sections and reset their states."
        self.file_select_button.config(state=tk.NORMAL, text="Selecteer puntenwolkbestand")
        self.point_amount_label.config(text="")
        self.file_label.config(text="Geen bestand geselecteerd")

        self.points_per_cm_entry.config(state=tk.DISABLED)
        Tooltip(self.points_per_cm_entry, "")
        self.point_density_button.config(state=tk.DISABLED, text="Pas puntdichtheid aan")
        Tooltip(self.point_density_button, "")
        self.point_density_result_label.config(text="")

        self.neighbour_amount_entry.config(state=tk.DISABLED)
        Tooltip(self.neighbour_amount_entry, "")
        self.std_ratio_entry.config(state=tk.DISABLED)
        Tooltip(self.std_ratio_entry, "")
        self.show_removed_points_checkbox.config(state=tk.DISABLED)
        Tooltip(self.show_removed_points_checkbox, "")
        self.show_removed_points_var.set(False)
        self.preprocessing_button.config(state=tk.DISABLED, text="Start voorbewerking")
        Tooltip(self.preprocessing_button, "")
        self.preprocessing_result_label.config(text="")

        self.visualize_heightmap_checkbox.config(state=tk.DISABLED)
        Tooltip(self.visualize_heightmap_checkbox, "")
        self.visualize_heightmap_var.set(False)
        self.heightmap_button.config(state=tk.DISABLED, text="Maak hoogtekaart")
        Tooltip(self.heightmap_button, "")
        self.heightmap_result_label.config(text="Hoogtekaart niet gemaakt.")

        self.floor_alpha_value_entry.config(state=tk.DISABLED)
        Tooltip(self.floor_alpha_value_entry, "")
        self.floor_triangle_size_entry.config(state=tk.DISABLED)
        Tooltip(self.floor_triangle_size_entry, "")
        self.corner_distance_threshold_entry.config(state=tk.DISABLED)
        Tooltip(self.corner_distance_threshold_entry, "")
        self.floor_detection_button.config(state=tk.DISABLED, text="Detecteer vloergrens")
        Tooltip(self.floor_detection_button, "")
        self.floor_to_cityjson_button.config(state=tk.DISABLED, text="Vloer naar 2D CityJSON")
        Tooltip(self.floor_to_cityjson_button, "")
        self.floor_detection_result_label.config(text="Vloergrens niet gedetecteerd.")

        self.slice_height_entry.config(state=tk.DISABLED)
        Tooltip(self.slice_height_entry, "")
        self.roof_extraction_button.config(state=tk.DISABLED, text="Extraheer dakpunten")
        Tooltip(self.roof_extraction_button, "")
        self.roof_extraction_result_label.config(text="Dak niet geëxtraheerd.")

        self.roof_layers_entry.config(state=tk.DISABLED)
        Tooltip(self.roof_layers_entry, "")
        self.roof_layer_fatness_entry.config(state=tk.DISABLED)
        Tooltip(self.roof_layer_fatness_entry, "")
        self.roof_voxel_size_entry.config(state=tk.DISABLED)
        Tooltip(self.roof_voxel_size_entry, "")
        self.roof_angle_threshold_entry.config(state=tk.DISABLED)
        Tooltip(self.roof_angle_threshold_entry, "")
        self.roof_merge_radius_entry.config(state=tk.DISABLED)
        Tooltip(self.roof_merge_radius_entry, "")
        self.roof_division_button.config(state=tk.DISABLED, text="Verdeel dak")
        Tooltip(self.roof_division_button, "")
        self.roof_division_result_label.config(text="Dak niet verdeeld.")

        self.wall_search_radius_entry.config(state=tk.DISABLED)
        Tooltip(self.wall_search_radius_entry, "")
        self.wall_extraction_button.config(state=tk.DISABLED, text="Extraheer muren")
        Tooltip(self.wall_extraction_button, "")
        self.wall_extraction_result_label.config(text="Muren niet geëxtraheerd.")

        self.wall_layer_amount_entry.config(state=tk.DISABLED)
        Tooltip(self.wall_layer_amount_entry, "")
        self.wall_division_button.config(state=tk.DISABLED, text="Verdeel muren")
        Tooltip(self.wall_division_button, "")
        self.wall_division_result_label.config(text="Muren niet verdeeld.")

        self.xy_tolerance_entry.config(state=tk.DISABLED)
        Tooltip(self.xy_tolerance_entry, "")
        self.max_line_length_entry.config(state=tk.DISABLED)
        Tooltip(self.max_line_length_entry, "")
        self.pcd_to_lineset_button.config(state=tk.DISABLED, text="Converteer naar\nLineset")
        Tooltip(self.pcd_to_lineset_button, "")
        self.pcd_to_lineset_result_label.config(text="Lineset niet gemaakt.")

        self.lineset_to_mesh_button.config(state=tk.DISABLED, text="Converteer naar Mesh")
        Tooltip(self.lineset_to_mesh_button, "")
        self.lineset_to_mesh_result_label.config(text="Mesh niet gemaakt.")

        self.repair_mesh_button.config(state=tk.DISABLED, text="Repareer Mesh")
        Tooltip(self.repair_mesh_button, "")
        self.repair_mesh_result_label.config(text="Mesh niet gerepareerd.")

        self.cityjson_conversion_button.config(state=tk.DISABLED, text="Converteer naar CityJSON")
        Tooltip(self.cityjson_conversion_button, "")
        self.cityjson_conversion_result_label.config(text="Niet geconverteerd naar CityJSON.")

        self.view_button.config(state=tk.DISABLED)
        self.save_cityjson_button.config(state=tk.DISABLED)
        self.pointcloud = None
        self.file_label.config(text="Geen bestand geselecteerd")

    def enable_point_density_section(self):
        self.points_per_cm_entry.config(state=tk.NORMAL)
        Tooltip(self.points_per_cm_entry, "Het gewenste aantal punten per cm² in de puntenwolk.")
        self.point_density_button.config(state=tk.NORMAL)
        Tooltip(
            self.point_density_button,
            "Pas de dichtheid van de puntenwolk aan op basis van het opgegeven aantal punten per cm²."
        )

    def enable_preprocessing_section(self):
        self.neighbour_amount_entry.config(state=tk.NORMAL)
        Tooltip(
            self.neighbour_amount_entry,
            "Het aantal naburige punten dat wordt gebruikt om te bepalen of een punt als ruis wordt beschouwd."
        )
        self.std_ratio_entry.config(state=tk.NORMAL)
        Tooltip(
            self.std_ratio_entry,
            "De standaarddeviatieverhouding die wordt gebruikt om ruispunten te identificeren."
        )
        self.show_removed_points_checkbox.config(state=tk.NORMAL)
        Tooltip(
            self.show_removed_points_checkbox,
            "Als ingeschakeld, worden de verwijderde ruispunten weergegeven in de visualisatie als rode punten."
        )
        self.preprocessing_button.config(state=tk.NORMAL, text="Start voorbewerking")
        Tooltip(
            self.preprocessing_button,
            "Start het voorbewerkingsproces om ruis te verwijderen en de puntenwolk te optimaliseren."
        )

    def enable_heightmap_section(self):
        self.heightmap_button.config(state=tk.NORMAL, text="Maak hoogtekaart")
        Tooltip(
            self.heightmap_button,
            "Maak een hoogtekaart op basis van de puntenwolk."
        )
        self.visualize_heightmap_checkbox.config(state=tk.NORMAL)
        Tooltip(
            self.visualize_heightmap_checkbox,
            "Visualiseer de gemaakte hoogtekaart."
        )

    def enable_floor_detection_section(self):
        self.floor_detection_button.config(state=tk.NORMAL, text="Detecteer vloergrens")
        Tooltip(
            self.floor_detection_button,
            "Detecteer de grens van de vloer in de puntenwolk."
        )
        self.floor_alpha_value_entry.config(state=tk.NORMAL)
        Tooltip(
            self.floor_alpha_value_entry,
            "De alpha-waarde die wordt gebruikt bij de vloergrensdetectie."
        )
        self.floor_triangle_size_entry.config(state=tk.NORMAL)
        Tooltip(
            self.floor_triangle_size_entry,
            "De grootte van de driehoeken die worden gebruikt bij de vloergrensdetectie."
        )
        self.corner_distance_threshold_entry.config(state=tk.NORMAL)
        Tooltip(
            self.corner_distance_threshold_entry,
            "De afstandsdrempel die wordt gebruikt om hoeken te identificeren bij de vloergrensdetectie."
        )

    def enable_floor_to_cityjson_section(self):
        self.floor_to_cityjson_button.config(state=tk.NORMAL, text="Vloer naar 2D CityJSON")
        Tooltip(
            self.floor_to_cityjson_button,
            "Converteer de gedetecteerde vloergrens naar een 2D CityJSON-bestand."
        )

    def enable_roof_extraction_section(self):
        self.roof_extraction_button.config(state=tk.NORMAL, text="Extraheer dakpunten")
        Tooltip(
            self.roof_extraction_button,
            "Extraheer dakpunten uit de puntenwolk op basis van de opgegeven snijlaaghoogte."
        )
        self.slice_height_entry.config(state=tk.NORMAL)
        Tooltip(
            self.slice_height_entry,
            "De hoogte van de snijlaag die wordt gebruikt om dakpunten te extraheren."
        )

    def enable_roof_division_section(self):
        self.roof_division_button.config(state=tk.NORMAL, text="Verdeel dak")
        Tooltip(
            self.roof_division_button,
            "Verdeel het dak in lagen op basis van de opgegeven parameters."
        )
        self.roof_layers_entry.config(state=tk.NORMAL)
        Tooltip(
            self.roof_layers_entry,
            "Het aantal lagen waarin het dak moet worden verdeeld."
        )
        self.roof_layer_fatness_entry.config(state=tk.NORMAL)
        Tooltip(
            self.roof_layer_fatness_entry,
            "De dikte van elke daklaag bij de verdeling."
        )
        self.roof_voxel_size_entry.config(state=tk.NORMAL)
        Tooltip(
            self.roof_voxel_size_entry,
            "De voxelgrootte die wordt gebruikt bij de dakverdeling."
        )
        self.roof_angle_threshold_entry.config(state=tk.NORMAL)
        Tooltip(
            self.roof_angle_threshold_entry,
            "De hoekdrempel die wordt gebruikt om dakvlakken te identificeren."
        )
        self.roof_merge_radius_entry.config(state=tk.NORMAL)
        Tooltip(
            self.roof_merge_radius_entry,
            "De koppelradius die wordt gebruikt om nabijgelegen dakvlakken samen te voegen."
        )

    def enable_wall_extraction_section(self):
        self.wall_extraction_button.config(state=tk.NORMAL, text="Extraheer muren")
        Tooltip(
            self.wall_extraction_button,
            "Extraheer muurpunten uit de puntenwolk op basis van de opgegeven zoekradius."
        )
        self.wall_search_radius_entry.config(state=tk.NORMAL)
        Tooltip(
            self.wall_search_radius_entry,
            "De zoekradius die wordt gebruikt om muurpunten te identificeren."
        )

    def enable_wall_division_section(self):
        self.wall_division_button.config(state=tk.NORMAL, text="Verdeel muren")
        Tooltip(
            self.wall_division_button,
            "Verdeel de muren in lagen op basis van het opgegeven aantal lagen."
        )
        self.wall_layer_amount_entry.config(state=tk.NORMAL)
        Tooltip(
            self.wall_layer_amount_entry,
            "Het aantal lagen waarin de muren moeten worden verdeeld."
        )

    def enable_pcd_to_lineset_section(self):
        self.pcd_to_lineset_button.config(state=tk.NORMAL, text="Converteer naar Lineset")
        Tooltip(
            self.pcd_to_lineset_button,
            "Converteer de puntenwolk naar een Lineset op basis van de opgegeven toleranties."
        )
        self.xy_tolerance_entry.config(state=tk.NORMAL)
        Tooltip(
            self.xy_tolerance_entry,
            "De XY-tolerantie die wordt gebruikt bij de conversie naar Lineset."
        )
        self.max_line_length_entry.config(state=tk.NORMAL)
        Tooltip(
            self.max_line_length_entry,
            "De maximale lijnlengte die wordt toegestaan bij de conversie naar Lineset."
        )

    def enable_lineset_to_mesh_section(self):
        self.lineset_to_mesh_button.config(state=tk.NORMAL, text="Converteer naar Mesh")
        Tooltip(
            self.lineset_to_mesh_button,
            "Converteer de Lineset naar een 3D Mesh."
        )
        self.lineset_to_mesh_result_label.config(text="Mesh niet gemaakt.")

    def enable_repair_mesh_section(self):
        self.repair_mesh_button.config(state=tk.NORMAL, text="Repareer Mesh")
        Tooltip(
            self.repair_mesh_button,
            "Repareer de 3D Mesh om eventuele fouten of gaten te herstellen."
        )
        self.repair_mesh_result_label.config(text="Mesh niet gerepareerd.")

    def enable_cityjson_conversion_section(self):
        self.cityjson_conversion_button.config(state=tk.NORMAL, text="Converteer naar CityJSON")
        Tooltip(
            self.cityjson_conversion_button,
            "Converteer de 3D Mesh naar een CityJSON-bestand."
        )
        self.cityjson_conversion_result_label.config(text="Niet geconverteerd naar CityJSON.")

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
        # Not sure if this even works
        config = configparser.ConfigParser()
        presets_file = 'presets.ini'  # Think about making this user definable later
        current_dir = os.path.dirname(os.path.abspath(__file__))
        presets_file = os.path.join(current_dir, presets_file)

        config.read(presets_file)

        # Check if there is an presets_file found. If not give a tkinter warning and skip loading presets.
        if not os.path.isfile(presets_file):
            tk.messagebox.showwarning("Waarschuwing", f"Bestand met presets niet gevonden: {presets_file}")
            return

        try:
            # Temporarily enable all entry fields, insert values if they exist, then disable them again
            entries_and_keys = [
                (self.points_per_cm_entry, 'points_per_cm'),
                (self.neighbour_amount_entry, 'neighbour_amount'),
                (self.std_ratio_entry, 'std_ratio'),
                (self.floor_alpha_value_entry, 'alpha_value'),
                (self.floor_triangle_size_entry, 'triangle_size'),
                (self.corner_distance_threshold_entry, 'distance_threshold'),
                (self.slice_height_entry, 'slice_height'),
                (self.roof_layers_entry, 'roof_layers'),
                (self.roof_layer_fatness_entry, 'roof_layer_fatness'),
                (self.roof_voxel_size_entry, 'roof_voxel_size'),
                (self.roof_angle_threshold_entry, 'angle_threshold'),
                (self.roof_merge_radius_entry, 'merge_radius'),
                (self.wall_search_radius_entry, 'wall_search_radius'),
                (self.wall_layer_amount_entry, 'wall_layer_amount'),
                (self.xy_tolerance_entry, 'xy_tolerance'),
                (self.max_line_length_entry, 'max_line_length')
            ]

            for entry, key in entries_and_keys:
                # Only insert value if it exists in the config file
                if config.has_option('Settings', key):
                    value = config.get('Settings', key)
                    entry.config(state=tk.NORMAL)       # Enable temporarily
                    entry.delete(0, tk.END)             # Clear any existing content
                    entry.insert(0, value)              # Insert the preset value
                    entry.config(state=tk.DISABLED)     # Disable again
                # If no value exists, leave the field empty (don't insert anything)

        except Exception as e:
            print(f"Fout bij het laden van presets: {e}")

    def on_close(self):
        # Perform any cleanup or final actions before closing
        if messagebox.askokcancel("Afsluiten", "Weet je zeker dat je wilt afsluiten?"):
            self.root.quit()  # Exit the Tkinter main loop
            exit()


def main():
    root = tk.Tk()
    app = App(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
