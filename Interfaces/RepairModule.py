# New interface 4 sections
# Grid division
# Plane detection in grid
# Repair point cloud with planes found
# Save repaired point cloud

import sys
import os
import tkinter as tk
import threading

# Insert the project path for importing modules
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

from Source.gridRansacModule import divide_pointcloud_into_grid, walk_through_grid
from Source.pointCloudEditor import open_point_cloud_editor
from Source.shapeUtils import repair_point_cloud_module, transform_mesh_to_pcd
from Source.fileHandler import save_pcd_as_las


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
    def __init__(self, root, point_cloud_data=None):
        self.root = root
        self.root.title("Repair Module")
        self.root.update_idletasks()  # Ensure the window is updated before setting geometry
        self.root.geometry("")  # Let geometry be determined by content
        self.root.resizable(False, False)  # Allow horizontal resizing
        self.root.iconbitmap("Source/support_files/logo.ico")

        self.point_cloud_data = point_cloud_data
        self.grid = None
        self.plane_pointcloud = None
        self.mesh = None

        # Initialize the parameters with default values
        self.grid_size = tk.DoubleVar(value=1.0)
        self.grid_overlap = tk.IntVar(value=0)
        self.min_cell_size = tk.IntVar(value=250)
        self.min_plane_size = tk.IntVar(value=500)
        self.kdtree_max_nn = tk.IntVar(value=100)
        self.depth = tk.IntVar(value=13)
        self.quantile_value = tk.DoubleVar(value=0.01)
        self.scale = tk.DoubleVar(value=2.2)

        self.entry_width = 16
        self.button_width = 20

        self.create_window()

    def validate_integer_input(self, value):
        return value.isdigit()

    def validate_float_input(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    def start_grid_division_thread(self):
        # Check if the grid size and overlap are valid
        if not self.validate_float_input(self.grid_size.get()):
            print("Grid size must be a float.")
            return

        self.disable_section(self.grid_division_button, "Grid division in progress...")
        threading.Thread(target=self.grid_division).start()

    def start_plane_detection_thread(self):
        self.disable_section(self.plane_detection_button, "Plane detection in progress...")
        threading.Thread(target=self.plane_detection).start()

    def start_repair_point_cloud_thread(self):
        self.disable_section(self.repair_point_cloud_button, "Point cloud repair in progress...")
        threading.Thread(target=self.repair_point_cloud).start()

    def grid_division(self):
        self.grid, grid_dimensions = divide_pointcloud_into_grid(
            self.point_cloud_data,
            self.grid_size.get(),
            self.grid_overlap.get(),
            give_grid_size=True
        )
        self.grid_devision_result_label.config(text=f"{len(self.grid)} cells created.\nGrid dimensions: {grid_dimensions}")
        self.enable_plane_detection_section()
        self.grid_division_button.config(state=tk.NORMAL, text="Start Grid Division")

    def plane_detection(self):
        try:
            self.plane_pointcloud = walk_through_grid(
                self.point_cloud_data, self.grid, self.min_cell_size.get(), self.min_plane_size.get()
            )
            self.plane_detection_result_label.config(
                text=f"Plane detection completed.\n{len(self.plane_pointcloud)} planes found."
            )
            self.enable_repair_section()
        except Exception as e:
            print(e)
            self.plane_detection_result_label.config(text="Error during plane detection.")
            self.plane_detection_button.config(state=tk.NORMAL, text="Start Plane Detection")
            self.enable_repair_section()

    def repair_point_cloud(self):
        self.mesh = repair_point_cloud_module(
            self.plane_pointcloud,
            visualize=False,
            kdtree_max_nn=self.kdtree_max_nn.get(),
            depth=self.depth.get(),
            quantile_value=self.quantile_value.get(),
            scale=self.scale.get()
        )
        self.repair_point_cloud_result_label.config(text="Point cloud repair completed.")
        self.enable_save_section()

    def save_repaired_point_cloud(self):
        if self.mesh is not None:
            transformed_pcd = transform_mesh_to_pcd(self.mesh, self.point_cloud_data)
            save_pcd_as_las(transformed_pcd)
            print("Repaired point cloud saved.")
        else:
            print("Repair must be completed first.")

    def create_window(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Grid Division Frame
        grid_division_frame = tk.LabelFrame(main_frame, text="Grid Division")
        grid_division_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for i in range(3):
            grid_division_frame.grid_columnconfigure(i, weight=1)

        tk.Label(grid_division_frame, text="Grid Size").grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        tk.Entry(grid_division_frame, textvariable=self.grid_size, width=self.entry_width).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Label(grid_division_frame, text="Grid Overlap").grid(
            row=1, column=0, padx=5, pady=5, sticky="ew"
        )
        tk.Entry(grid_division_frame, textvariable=self.grid_overlap, width=self.entry_width).grid(
            row=1, column=1, padx=5, pady=5, sticky="ew"
        )

        self.grid_division_button = tk.Button(
            grid_division_frame, text="Start Grid Division", width=self.button_width, command=self.start_grid_division_thread
        )
        self.grid_division_button.grid(
            row=1, column=2, padx=5, pady=5, sticky="ew"
        )

        self.grid_devision_result_label = tk.Label(
            grid_division_frame, text="Grid not divided yet.", anchor="w"
        )
        self.grid_devision_result_label.grid(
            row=0, column=2, padx=5, pady=5, sticky="ew"
        )

        # Plane Detection Frame
        plane_detection_frame = tk.LabelFrame(main_frame, text="Plane Detection")
        plane_detection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for i in range(3):
            plane_detection_frame.grid_columnconfigure(i, weight=1)

        tk.Label(plane_detection_frame, text="Min Cell Size").grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        self.min_cell_size_entry = tk.Entry(
            plane_detection_frame, textvariable=self.min_cell_size, width=self.entry_width, state=tk.DISABLED
        )
        self.min_cell_size_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Label(plane_detection_frame, text="Min Plane Size").grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        self.min_plane_size_entry = tk.Entry(
            plane_detection_frame, textvariable=self.min_plane_size, width=self.entry_width, state=tk.DISABLED
        )
        self.min_plane_size_entry.grid(
            row=1, column=1, padx=5, pady=5, sticky="ew"
        )

        self.plane_detection_button = tk.Button(
            plane_detection_frame,
            text="Start Plane Detection",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_plane_detection_thread
        )
        self.plane_detection_button.grid(
            row=1, column=2, padx=5, pady=5, sticky="ew"
        )

        self.plane_detection_result_label = tk.Label(
            plane_detection_frame, text="Planes not detected yet.", anchor="w"
        )
        self.plane_detection_result_label.grid(
            row=0, column=2, padx=5, pady=5, sticky="ew"
        )

        # Repair Point Cloud Frame
        repair_point_cloud_frame = tk.LabelFrame(main_frame, text="Repair Point Cloud")
        repair_point_cloud_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for i in range(4):
            repair_point_cloud_frame.grid_columnconfigure(i, weight=1)

        tk.Label(repair_point_cloud_frame, text="Kdtree Max NN").grid(
            row=0, column=0, padx=5, pady=5, sticky="ew"
        )
        self.kdtree_max_nn_entry = tk.Entry(
            repair_point_cloud_frame, textvariable=self.kdtree_max_nn, width=self.entry_width, state=tk.DISABLED
        )
        self.kdtree_max_nn_entry.grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Label(repair_point_cloud_frame, text="Depth").grid(
            row=1, column=0, padx=5, pady=5, sticky="ew"
        )
        self.depth_entry = tk.Entry(repair_point_cloud_frame, textvariable=self.depth, width=self.entry_width, state=tk.DISABLED)
        self.depth_entry.grid(
            row=1, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Label(repair_point_cloud_frame, text="Quantile Value").grid(
            row=2, column=0, padx=5, pady=5, sticky="ew"
        )
        self.quantile_value_entry = tk.Entry(
            repair_point_cloud_frame, textvariable=self.quantile_value, width=self.entry_width, state=tk.DISABLED
        )
        self.quantile_value_entry.grid(
            row=2, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Label(repair_point_cloud_frame, text="Scale").grid(
            row=0, column=2, padx=5, pady=5, sticky="ew"
        )
        self.scale_entry = tk.Entry(repair_point_cloud_frame, textvariable=self.scale, width=self.entry_width, state=tk.DISABLED)
        self.scale_entry.grid(
            row=0, column=3, padx=5, pady=5, sticky="ew"
        )

        self.repair_point_cloud_button = tk.Button(
            repair_point_cloud_frame,
            text="Start Point Cloud Repair",
            width=self.button_width,
            state=tk.DISABLED,
            command=self.start_repair_point_cloud_thread
        )
        self.repair_point_cloud_button.grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        self.repair_point_cloud_result_label = tk.Label(
            repair_point_cloud_frame,
            text="Point cloud not repaired yet.",
            anchor="w"
        )
        self.repair_point_cloud_result_label.grid(row=2, column=2, columnspan=2, padx=5, pady=5, sticky="ew")

        # Save Repaired Point Cloud Frame
        save_frame = tk.LabelFrame(main_frame, text="Save Repaired Point Cloud")
        save_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        for i in range(2):
            save_frame.grid_columnconfigure(i, weight=1)

        self.save_button = tk.Button(
            save_frame,
            text="Save Repaired Point Cloud",
            width=self.button_width,
            height=2,
            state=tk.DISABLED,
            command=self.save_repaired_point_cloud)
        self.save_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.show_button = tk.Button(
            save_frame,
            text="Show Repaired Point Cloud",
            width=self.button_width,
            height=2,
            state=tk.DISABLED,
            command=lambda: open_point_cloud_editor(self.mesh, False) if self.mesh else print("No repaired point cloud to show."))
        self.show_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Exit Button
        exit_button = tk.Button(main_frame, text="Back", width=self.button_width, command=self.root.destroy)
        exit_button.pack(pady=10, fill=tk.X)

        self.root.bind("<Escape>", lambda event: exit_button.invoke())

        # Make the main window's column expand
        self.root.grid_columnconfigure(0, weight=1)

    def disable_section(self, button, label_text):
        button.config(state=tk.DISABLED, text=label_text)

    def enable_plane_detection_section(self):
        self.plane_detection_button.config(state=tk.NORMAL, text="Start Plane Detection")
        self.min_cell_size_entry.config(state=tk.NORMAL)
        self.min_plane_size_entry.config(state=tk.NORMAL)

    def enable_repair_section(self):
        self.repair_point_cloud_button.config(state=tk.NORMAL, text="Start Point Cloud Repair")
        self.kdtree_max_nn_entry.config(state=tk.NORMAL)
        self.depth_entry.config(state=tk.NORMAL)
        self.quantile_value_entry.config(state=tk.NORMAL)
        self.scale_entry.config(state=tk.NORMAL)

    def enable_save_section(self):
        self.save_button.config(state=tk.NORMAL)
        self.show_button.config(state=tk.NORMAL)


def main():
    root = tk.Tk()
    app = App(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()


# # Divide the pointcloud into a 3d grid
# grid = grm.divide_pointcloud_into_grid(pcd_stat, 1, 0)

# plane_pointcloud = grm.walk_through_grid(pcd_stat, grid, 250, 500)

# open_point_cloud_editor(plane_pointcloud, False)

# # Repair the point cloud with the planes found.
# # With these parameters, just try higher NN and depth and lower quantile value (and maybe scale)
# mesh = repair_point_cloud_module(plane_pointcloud, visualize=True, kdtree_max_nn=100, depth=13, quantile_value=0.01, scale=2.2)

# # Transform the mesh back to a point cloud.
# transformed_pcd = transform_mesh_to_pcd(mesh, plane_pointcloud)

# open_point_cloud_editor(transformed_pcd, False)
