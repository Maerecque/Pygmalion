# Dear me I ****** up with a merge conflict, lots of love from your past self.

# Ps I'm sorry for the language, I'm just a bit frustrated with the merge conflict and want to have weekend. I hope you can forgive me. KYS <3

import tkinter as tk
from tkinter import messagebox, ttk
import os
import sys
import open3d as o3d

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

# Import the necessary functions (assuming these functions are properly defined in your source)
from Source.gridRansacModule import divide_pointcloud_into_grid, walk_through_grid
from Source.shapeUtils import repair_point_cloud_module


class App:
    def __init__(self, root, point_cloud_data=None, point_cloud_path=None):
        self.root = root
        self.root.title("3D Processing Module")
        self.root.update_idletasks()
        self.root.geometry("")  # Let geometry be determined by content
        self.root.resizable(False, False)  # Allow horizontal resizing
        self.root.iconbitmap("Source\\support_files\\logo.ico")

        self.point_cloud_data = point_cloud_data
        self.point_cloud_path = point_cloud_path

        self.validate_int = self.root.register(self.validate_integer)
        self.validate_flt = self.root.register(self.validate_float)

        # Common width for entry fields
        self.entry_width = 18

        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.create_widgets()

    def create_widgets(self):
        # Grid Division Section
        grid_division_frame = ttk.LabelFrame(self.root, text="Grid division")
        grid_division_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        grid_division_frame.grid_columnconfigure(0, weight=1)
        grid_division_frame.grid_columnconfigure(1, weight=1)

        tk.Label(grid_division_frame, text="Grid size (float):", anchor="w").grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        self.grid_size = tk.Entry(grid_division_frame, width=self.entry_width, validate="key", validatecommand=(self.validate_flt, "%P"))
        self.grid_size.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(grid_division_frame, text="Overlap (integer):", anchor="w").grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        self.overlap = tk.Entry(grid_division_frame, width=self.entry_width, validate="key", validatecommand=(self.validate_int, "%P"))
        self.overlap.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.result_label = tk.Label(grid_division_frame, text="Result will be displayed here.", anchor="w")
        self.result_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.grid_division_start_button = tk.Button(grid_division_frame, text="Start", command=self.grid_division_start)
        self.grid_division_start_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Grid Walkthrough Section
        grid_walkthrough_frame = ttk.LabelFrame(self.root, text="Grid walkthrough")
        grid_walkthrough_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        grid_walkthrough_frame.grid_columnconfigure(0, weight=1)
        grid_walkthrough_frame.grid_columnconfigure(1, weight=1)

        tk.Label(grid_walkthrough_frame, text="Min. cell size:", anchor="w").grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        self.min_cell_size = tk.Entry(grid_walkthrough_frame, width=self.entry_width)
        self.min_cell_size.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(grid_walkthrough_frame, text="Min. plane size:", anchor="w").grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        self.min_plane_size = tk.Entry(grid_walkthrough_frame, width=self.entry_width)
        self.min_plane_size.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.grid_walkthrough_show_button = tk.Button(grid_walkthrough_frame, text="Show Result", command=self.grid_walkthrough_show)
        self.grid_walkthrough_show_button.grid(row=2, column=0, padx=5, pady=5, sticky="ew")
        self.grid_walkthrough_start_button = tk.Button(grid_walkthrough_frame, text="Start", command=self.grid_walkthrough_start)
        self.grid_walkthrough_start_button.grid(row=2, column=1, padx=5, pady=5, sticky="ew")

        # Pointcloud Repair Section
        pointcloud_repair_frame = ttk.LabelFrame(self.root, text="Pointcloud repair")
        pointcloud_repair_frame.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        pointcloud_repair_frame.grid_columnconfigure(0, weight=1)
        pointcloud_repair_frame.grid_columnconfigure(1, weight=1)

        tk.Label(pointcloud_repair_frame, text="KdTree NN:", anchor="w").grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        self.kdtree_nn = tk.Entry(pointcloud_repair_frame, width=self.entry_width)
        self.kdtree_nn.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(pointcloud_repair_frame, text="Depth:", anchor="w").grid(row=1, column=0, padx=5, pady=2, sticky="ew")
        self.depth = tk.Entry(pointcloud_repair_frame, width=self.entry_width)
        self.depth.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(pointcloud_repair_frame, text="Quantile value (float):", anchor="w").grid(row=2, column=0, padx=5, pady=2, sticky="ew")
        self.quantile_value = tk.Entry(pointcloud_repair_frame, width=self.entry_width)
        self.quantile_value.grid(row=2, column=1, padx=5, pady=2, sticky="ew")

        tk.Label(pointcloud_repair_frame, text="Scale (float):", anchor="w").grid(row=3, column=0, padx=5, pady=2, sticky="ew")
        self.scale = tk.Entry(pointcloud_repair_frame, width=self.entry_width)
        self.scale.grid(row=3, column=1, padx=5, pady=2, sticky="ew")

        self.pointcloud_show_button = tk.Button(pointcloud_repair_frame, text="Show Result", command=self.pointcloud_repair_show)
        self.pointcloud_show_button.grid(row=4, column=0, padx=5, pady=5, sticky="ew")
        self.pointcloud_start_button = tk.Button(pointcloud_repair_frame, text="Start", command=self.pointcloud_repair_start)
        self.pointcloud_start_button.grid(row=4, column=1, padx=5, pady=5, sticky="ew")

        # Save Button at the Bottom
        self.save_button = tk.Button(self.root, text="Save", height=2, command=self.save)
        self.save_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

        # Make the main window's column expand
        self.root.grid_columnconfigure(0, weight=1)

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

    def validate_fields(self):
        # Create a list to store the invalid fields
        invalid_fields = []

        # Reset the background color of all entry fields
        for entry in self.root.winfo_children():
            if isinstance(entry, tk.Entry):
                entry.config(bg="white")

        fields = {
            "Grid size": self.grid_size.get().replace(".", "").isdigit(),
            "Overlap": self.overlap.get().isdigit(),
            "Min. cell size": self.min_cell_size.get().isdigit(),
            "Min. plane size": self.min_plane_size.get().isdigit(),
            "KdTree NN": self.kdtree_nn.get().isdigit(),
            "Depth": self.depth.get().isdigit(),
            "Quantile value": self.quantile_value.get().replace(".", "").isdigit(),
            "Scale": self.scale.get().replace(".", "").isdigit()
        }
        invalid_fields = [field for field, valid in fields.items() if not valid]

        # Make the bg of the valid fields green
        for field in [field for field, valid in fields.items() if valid]:
            getattr(self, f"{field.lower().replace(' ', '_')}_entry").config(bg="white")

        # If there are invalid fields, show a popup with the field names
        if invalid_fields:
            # Highlight the invalid fields
            for field, valid in fields.items():
                if not valid:
                    getattr(self, f"{field.lower().replace(' ', '_')}_entry").config(bg="red")

            # Show a popup with the invalid fields
            self.show_message(
                "Invalid Fields",
                f"The following fields are invalid or empty: {', '.join(invalid_fields)}",
                "error"
            )

            # Focus on the first invalid field
            getattr(self, f"{invalid_fields[0].lower().replace(' ', '_')}_entry").focus()

            return False

        return True

    def grid_division_start(self):
        print("Grid division started")

    def grid_walkthrough_show(self):
        print("Grid walkthrough result shown")

    def grid_walkthrough_start(self):
        print("Grid walkthrough started")

    def pointcloud_repair_show(self):
        print("Pointcloud repair result shown")

    def pointcloud_repair_start(self):
        print("Pointcloud repair started")

    def save(self):
        print("Save button clicked")

    def on_close(self):
        # Perform any cleanup or final actions before closing
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()  # Exit the Tkinter main loop
            self.root.destroy()  # Close the Tkinter window
            exit()


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


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
