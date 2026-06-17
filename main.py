import configparser
import json
import locale
import numpy as np
import os
from random import randint as KernelMan
import sys
import re
import threading
import time
import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *  # noqa: F401, F403
from tqdm import tqdm

# Fix for PyInstaller windowed mode - redirect stdout/stderr if they're None
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

locale.setlocale(locale.LC_ALL, 'nl_NL.UTF-8')


def resource_path(relative_path):
    """Return the absolute path to a resource, compatible with PyInstaller's --onefile mode."""
    base = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    return os.path.join(base, relative_path)


def get_application_version():
    """Read the application version from the version_info.txt file."""
    version_file_path = resource_path(os.path.join("Source", "support_files", "version_info.txt"))
    try:
        if not os.path.exists(version_file_path):
            return "Unknown"
        with open(version_file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Zoek naar StringStruct('FileVersion', 'X.X.X')
            match = re.search(r"StringStruct\(\s*'FileVersion'\s*,\s*'([^']+)'\s*\)", content)
            if match:
                return match.group(1)
    except Exception:
        return "Unknown"


# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))

import open3d as o3d  # noqa: E402 — must come after sys.path setup
from Source.boundaryScript import expand_boundary
from Source.fileHandler import (  # noqa: F401
    get_file_path,
    readout_LAS_file,
    readout_e57_file,
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
from Source.pointCloudEditor import (  # noqa: F401
    open_point_cloud_editor as opce,
    open_mesh_and_lineset_viewer as omalv
)
from Source.roofTools import slice_roof_up
from Source.wallTools import (
    extract_wall_points,
    define_min_height_roof,
    connect_vertically_aligned_points,
    connect_vertically_aligned_points2,
    divide_wall_into_layers
)

# ── constants ────────────────────────────────────────────────────────────────
DARK_THEME = "darkly"
LIGHT_THEME = "flatly"

STEP_LABELS = [
    "Bestand selecteren",      # 1
    "Puntdichtheid",           # 2
    "Ruis verwijderen",        # 3
    "Hoogtekaart",             # 4
    "Vloergrens detectie",     # 5
    "Vloer uitbreiden",        # 6
    "Vloer → 2D CityJSON",     # 7
    "Dakextractie",            # 8
    "Dakverdeling",            # 9
    "Wandextractie",           # 10
    "Wandverdeling",           # 11
    "PCD → Lineset",           # 12
    "Lineset → Mesh",          # 13
    "Mesh reparatie",          # 14
    "CityJSON conversie",      # 15
]

# step states
PENDING  = "pending"   # noqa: E221
ACTIVE   = "active"    # noqa: E221
COMPLETE = "complete"  # noqa: E221
ERROR    = "error"     # noqa: E221
OPTIONAL = "info"  # noqa: E221


# ── Tooltip ───────────────────────────────────────────────────────────────────
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

    # ── public ──────────────────────────────────────────────────────────
    def reset(self):
        self.cancel_all()
        self.destroy_tooltip()

    # ── event handlers ───────────────────────────────────────────────────
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
            self.timer_id = self.widget.after(500, self.show_tooltip)
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

            # Detect current style/theme for colors
            try:
                style = ttk.Style.instance
                bg = style.colors.dark if style else "#2b2b2b"
                fg = style.colors.light if style else "#ffffff"
            except Exception:
                bg = "#2b2b2b"
                fg = "#f0f0f0"

            label = tk.Label(
                self.tipwindow,
                text=self.text,
                justify="left",
                background=bg,
                foreground=fg,
                relief="flat",
                borderwidth=0,
                font=("Segoe UI", 9),
                padx=8,
                pady=4,
            )
            label.pack()
            self.check_position()
        except Exception:
            self.destroy_tooltip()

    def check_position(self):
        if not self.tipwindow:
            self.check_id = None
            return
        try:
            mx = self.widget.winfo_pointerx()
            my = self.widget.winfo_pointery()
            wx = self.widget.winfo_rootx()
            wy = self.widget.winfo_rooty()
            ww = self.widget.winfo_width()
            wh = self.widget.winfo_height()
            if not (wx <= mx <= wx + ww and wy <= my <= wy + wh):
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


# ── App ───────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: ttk.Window, point_cloud_data=None, point_cloud_path=None):
        self.root = root
        current_version = get_application_version()
        self.root.title(f"Pygmalion - v{current_version}")
        self.root.resizable(True, True)
        self.root.minsize(860, 645)

        try:
            self.root.iconbitmap(resource_path(os.path.join("Source", "support_files", "logo.ico")))
        except Exception:
            pass

        # Theme state
        self._dark_mode = True

        # Point cloud storage
        self.point_cloud_data = point_cloud_data
        self.point_cloud_path = point_cloud_path

        # Tooltip tracking
        self.tooltips: list[Tooltip] = []

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

        # Sidebar step state tracking: list index 0 = step 1
        self._step_states = [PENDING] * 15
        self._step_rows: list[dict] = []   # [{frame, num_lbl, name_lbl, icon_lbl}, ...]
        self._current_step = 0              # 1-based, 0 = none

        # Register validation
        self.validate_int = self.root.register(self.validate_integer)
        self.validate_flt = self.root.register(self.validate_float)

        # Build UI
        self._build_ui()

        # Setup button cursors
        self._setup_button_cursors()

        # Auto-size window to fit content, then center on screen
        self.root.update_idletasks()
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()
        w = max(req_w, 860)
        h = max(req_h, 645)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # Load presets
        self.load_presets()

        # Key bindings
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # If point cloud data is provided, load it
        if self.point_cloud_data is not None and self.point_cloud_path is not None:
            self.load_point_cloud_data()

        # Periodic internal validation
        self._schedule_integrity_check()

    # ── UI builders ──────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_header()
        self._build_body()
        self._build_status_bar()

    def _setup_button_cursors(self):
        def on_btn_enter(event):
            try:
                btn = event.widget
                if str(btn.cget("state")) == "disabled":
                    btn.configure(cursor="arrow")
                else:
                    btn.configure(cursor="hand2")
            except Exception:
                pass

        self.root.bind_class("TButton", "<Enter>", on_btn_enter, add="+")

    def _build_header(self):
        hdr = ttk.Frame(self.root, padding=(12, 8), bootstyle="dark")
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.columnconfigure(1, weight=1)

        # Theme toggle
        self.theme_btn = ttk.Button(
            hdr, text="☀", width=3,
            bootstyle="secondary-outline",
            command=self._toggle_theme
        )
        self.theme_btn.grid(row=0, column=0, padx=(0, 12))
        self.add_tooltip(self.theme_btn, "Wissel tussen donker en licht thema.")

        # Title
        ttk.Label(
            hdr, text="Pygmalion CityJSON Generator",
            font=("Segoe UI", 14, "bold"),
            bootstyle="inverse-dark"
        ).grid(row=0, column=1, sticky="w")

        # Right-side actions
        btn_frame = ttk.Frame(hdr, bootstyle="dark")
        btn_frame.grid(row=0, column=2)

        self.reset_button = ttk.Button(
            btn_frame, text="↺  Reset", width=10,
            bootstyle="warning-outline",
            command=self.reset_application
        )
        self.reset_button.pack(side="left", padx=(0, 8))
        self.add_tooltip(self.reset_button, "Reset de applicatie naar de begintoestand.")

        ttk.Button(
            btn_frame, text="✕  Sluiten", width=10,
            bootstyle="danger-outline",
            command=self.on_close
        ).pack(side="left")

    def _build_body(self):
        body = ttk.Frame(self.root)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_content(body)

    def _build_sidebar(self, parent):
        sidebar_outer = ttk.Frame(parent, bootstyle="dark", width=275)
        sidebar_outer.grid(row=0, column=0, sticky="ns")
        sidebar_outer.pack_propagate(False)
        sidebar_outer.grid_propagate(False)

        # ── scrollable inner area ──
        canvas = tk.Canvas(sidebar_outer, bg=self._theme_color("dark"), highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_outer, orient="vertical", command=canvas.yview, bootstyle="dark-round")
        self._sidebar_inner = ttk.Frame(canvas, bootstyle="dark")

        def _on_inner_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            if self._sidebar_inner.winfo_reqheight() > canvas.winfo_height():
                scrollbar.pack(side="right", fill="y")
            else:
                scrollbar.pack_forget()

        self._sidebar_inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(self._sidebar_window, width=e.width))
        self._sidebar_window = canvas.create_window((0, 0), window=self._sidebar_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel to canvas — only scroll when content overflows
        def _on_mousewheel(e):
            if self._sidebar_inner.winfo_reqheight() > canvas.winfo_height():
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._sidebar_canvas = canvas

        # ── section label ──
        ttk.Label(
            self._sidebar_inner,
            text="PIPELINE",
            font=("Segoe UI", 9, "bold"),
            bootstyle="inverse-dark",
            padding=(14, 10, 0, 4)
        ).pack(fill="x")

        ttk.Separator(self._sidebar_inner, bootstyle="secondary").pack(fill="x", padx=10, pady=(0, 4))

        # ── step rows ──
        self._step_rows = []
        for i, label in enumerate(STEP_LABELS):
            row = self._make_sidebar_row(self._sidebar_inner, i + 1, label)
            self._step_rows.append(row)

        ttk.Separator(self._sidebar_inner, bootstyle="secondary").pack(fill="x", padx=10, pady=8)

        # ── View / Save at sidebar bottom ──
        self.view_button = ttk.Button(
            self._sidebar_inner,
            text="👁  Bekijk resultaat",
            state="disabled",
            bootstyle="info-outline",
            command=lambda: None,
        )
        self.view_button.pack(padx=8, pady=(0, 6), fill="x")
        self.add_tooltip(self.view_button, "Bekijk het huidige resultaat in een nieuw venster.")

        self.save_cityjson_button = ttk.Button(
            self._sidebar_inner,
            text="💾  Sla CityJSON op",
            state="disabled",
            bootstyle="success-outline",
            command=self.save_cityjson_file_step,
        )
        self.save_cityjson_button.pack(padx=8, pady=(0, 10), fill="x")
        self.add_tooltip(self.save_cityjson_button, "Sla het gegenereerde CityJSON-bestand op.")

    def _make_sidebar_row(self, parent, step_num: int, label: str) -> dict:
        frame = ttk.Frame(parent, bootstyle="dark", padding=(8, 4))
        frame.pack(fill="x")

        num_lbl = ttk.Label(
            frame,
            text=f"{step_num:02d}",
            font=("Segoe UI", 9, "bold"),
            bootstyle="inverse-dark",
            width=3
        )
        num_lbl.pack(side="left")

        name_lbl = ttk.Label(
            frame,
            text=label,
            font=("Segoe UI", 9),
            bootstyle="inverse-dark",
        )
        name_lbl.pack(side="left", padx=(6, 4), fill="x", expand=True)

        icon_lbl = ttk.Label(
            frame,
            text="○",
            font=("Segoe UI", 10),
            bootstyle="inverse-dark",
            width=2
        )
        icon_lbl.pack(side="right")

        # Local function for Hover ON (changes styles to indicate hover)
        def on_enter(e):
            # retrieve current cursor style based on step state; only allow pointer if actionable
            cursor_style = "hand2"
            state = self._step_states[step_num - 1]
            if state not in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
                cursor_style = "arrow"
            frame.configure(bootstyle="secondary", cursor=cursor_style)
            for lbl in (num_lbl, name_lbl, icon_lbl):
                lbl.configure(bootstyle="inverse-secondary", cursor=cursor_style)

        # Local function for Hover OFF (restores the correct styles based on the status)
        def on_leave(e):
            state = self._step_states[step_num - 1]

            # Determine the frame color (Note: change "info" to "secondary" if it better matches _update_sidebar_step)
            f_style = {
                PENDING: "dark",
                ACTIVE: "primary",
                COMPLETE: "success",
                ERROR: "danger",
                OPTIONAL: "info"
            }.get(state, "dark")

            # Determine the text/label color
            l_style = {
                PENDING: "inverse-dark",
                ACTIVE: "inverse-primary",
                COMPLETE: "inverse-success",
                ERROR: "inverse-danger",
                OPTIONAL: "inverse-info"
            }.get(state, "inverse-dark")

            frame.configure(bootstyle=f_style)
            for lbl in (num_lbl, name_lbl, icon_lbl):
                lbl.configure(bootstyle=l_style)

        # Click navigation and hover functionality binding to all widgets
        for widget in (frame, num_lbl, name_lbl, icon_lbl):
            widget.configure(cursor="hand2")
            widget.bind("<Button-1>", lambda e, n=step_num: self._sidebar_click(n))
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        return {"frame": frame, "num_lbl": num_lbl, "name_lbl": name_lbl, "icon_lbl": icon_lbl}

    def _sidebar_click(self, step_num: int):
        state = self._step_states[step_num - 1]
        if state in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.show_step(step_num)

    def _build_content(self, parent):
        self._content_frame = ttk.Frame(parent, padding=(16, 12))
        self._content_frame.grid(row=0, column=1, sticky="nsew")
        self._content_frame.columnconfigure(0, weight=1)
        self._content_frame.rowconfigure(0, weight=1)

        # Placeholder shown before first step
        self._show_welcome()

    def _show_welcome(self):
        self._clear_content()
        f = ttk.Frame(self._content_frame)
        f.grid(row=0, column=0, sticky="nsew")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(0, weight=1)

        inner = ttk.Frame(f)
        inner.grid(row=0, column=0)

        ttk.Label(inner, text="Bestandsselectie",
                  font=("Segoe UI", 18, "bold")).pack(pady=(0, 8))
        ttk.Label(inner,
                  text="Selecteer een puntenwolkbestand via stap 1 om te beginnen.",
                  font=("Segoe UI", 11), bootstyle="secondary").pack()
        ttk.Button(
            inner, text="📂  Selecteer puntenwolkbestand",
            bootstyle="primary", command=self.select_file, width=30
        ).pack(pady=20)

    def _build_status_bar(self):
        bar = ttk.Frame(self.root, padding=(12, 4), bootstyle="dark")
        bar.grid(row=2, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(
            bar, text="Gereed",
            font=("Segoe UI", 9),
            bootstyle="inverse-dark"
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress_bar = ttk.Progressbar(
            bar, mode="indeterminate", length=160, bootstyle="info-striped"
        )
        self.progress_bar.grid(row=0, column=1, sticky="e")
        self.progress_bar.grid_remove()  # hidden by default

    # ── step panel builders ──────────────────────────────────────────────────

    def show_step(self, step_num: int):
        self._clear_content()
        self._current_step = step_num
        builders = {
            1: self._build_step_1_panel,
            2: self._build_step_2_panel,
            3: self._build_step_3_panel,
            4: self._build_step_4_panel,
            5: self._build_step_5_panel,
            6: self._build_step_6_panel,
            7: self._build_step_7_panel,
            8: self._build_step_8_panel,
            9: self._build_step_9_panel,
            10: self._build_step_10_panel,
            11: self._build_step_11_panel,
            12: self._build_step_12_panel,
            13: self._build_step_13_panel,
            14: self._build_step_14_panel,
            15: self._build_step_15_panel,
        }
        builders[step_num]()

    def _clear_content(self):
        for child in self._content_frame.winfo_children():
            child.destroy()

    def _step_card(self, step_num: int, title: str) -> ttk.Frame:
        """Create and return the inner content frame for a step card."""
        outer = ttk.Frame(self._content_frame)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)

        # Step heading
        hdr = ttk.Frame(outer)
        hdr.pack(fill="x", pady=(0, 12))
        ttk.Label(
            hdr,
            text=f"Stap {step_num}  —  {title}",
            font=("Segoe UI", 14, "bold")
        ).pack(side="left")

        card = ttk.LabelFrame(outer, text="Parameters & Actie")
        card.pack(fill="both", expand=True, padx=2, pady=2)
        card.columnconfigure(1, weight=1)
        # Apply internal padding via a child frame
        inner = ttk.Frame(card, padding=16)
        inner.pack(fill="both", expand=True)
        inner.columnconfigure(1, weight=1)
        return inner

    def _field(
            self, parent, row: int, label: str, attr: str,
            validate_cmd=None, tooltip: str = "") -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 12))
        kwargs = dict(width=18, state="disabled")
        if validate_cmd:
            kwargs["validate"] = "key"
            kwargs["validatecommand"] = validate_cmd
        entry = ttk.Entry(parent, **kwargs)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        setattr(self, attr, entry)
        if tooltip:
            self.add_tooltip(entry, tooltip)
        return entry

    def _next_grid_row(self, parent) -> int:
        """Return the next available grid row in parent (max existing row + 1, min 20)."""
        slaves = parent.grid_slaves()
        if not slaves:
            return 20
        return max(int(w.grid_info().get("row", 0)) for w in slaves) + 1

    def _result_label(self, parent, attr: str, default_text: str = "") -> ttk.Label:
        row = self._next_grid_row(parent)
        lbl = ttk.Label(parent, text=default_text, font=("Segoe UI", 9), bootstyle="secondary")
        lbl.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        setattr(self, attr, lbl)
        return lbl

    def _action_btn(self, parent, text: str, attr: str, command,
                    bootstyle="primary", tooltip: str = "") -> ttk.Button:
        row = self._next_grid_row(parent)
        btn = ttk.Button(parent, text=text, command=command,
                         bootstyle=bootstyle, state="disabled", width=26)
        btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        setattr(self, attr, btn)
        if tooltip:
            self.add_tooltip(btn, tooltip)
        return btn

    # ── Step 1 — Bestand selecteren ──────────────────────────────────────────
    def _build_step_1_panel(self):
        card = self._step_card(1, "Bestand selecteren")

        ttk.Label(card, text="Selecteer een .las, .laz of .e57 puntenwolkbestand.").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        self.file_select_button = ttk.Button(
            card, text="📂  Selecteer puntenwolkbestand",
            bootstyle="primary", command=self.select_file, width=32
        )
        self.file_select_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.add_tooltip(self.file_select_button, "Selecteer een .las, .laz of .e57 puntenwolkbestand om te beginnen.")

        self.file_label = ttk.Label(card, text="Geen bestand geselecteerd",
                                    bootstyle="secondary", wraplength=550, justify="left")
        self.file_label.grid(row=2, column=0, columnspan=2, sticky="w")

        self.point_amount_label = ttk.Label(card, text="", font=("Segoe UI", 9, "bold"))
        self.point_amount_label.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # Restore state if a file was already loaded
        if hasattr(self, 'point_cloud_path') and self.point_cloud_path:
            self.file_select_button.configure(text="\U0001f4c2\u2002 Bestand wijzigen")
            self.file_label.configure(
                text=f"Geselecteerd: {os.path.basename(self.point_cloud_path)}",
                bootstyle="default"
            )
            if hasattr(self, 'point_cloud_data') and self.point_cloud_data is not None:
                self.point_amount_label.configure(
                    text=f"Punten: {len(self.point_cloud_data.points):n}"
                )

    # ── Step 2 — Puntdichtheid ───────────────────────────────────────────────
    def _build_step_2_panel(self):
        card = self._step_card(2, "Puntdichtheid aanpassen")
        self._field(
            card, 0, "Punten per cm²", "points_per_cm_entry", (self.validate_flt, '%P'),
            "Het gewenste aantal punten per cm² in de puntenwolk."
        )
        self._action_btn(
            card, "Pas puntdichtheid aan", "point_density_button",
            self.start_alter_point_density_thread,
            tooltip="Pas de dichtheid van de puntenwolk aan."
        )
        self._result_label(card, "point_density_result_label")
        if self._step_states[1] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.points_per_cm_entry.configure(state="normal")
            self.point_density_button.configure(state="normal")
            self._load_preset_into("points_per_cm_entry", "points_per_cm")

    # ── Step 3 — Ruis verwijderen ────────────────────────────────────────────
    def _build_step_3_panel(self):
        card = self._step_card(3, "Ruis verwijderen")
        self._field(
            card, 0, "Aantal buren", "neighbour_amount_entry", (self.validate_int, '%P'),
            "Aantal naburige punten voor ruisbeoordeling. \n Hoe meer buren, hoe grondiger de beoordeling, maar ook hoe langer het proces duurt."  # noqa: E501
        )
        self._field(
            card, 1, "Std ratio", "std_ratio_entry", (self.validate_flt, '%P'),
            "Standaarddeviatieverhouding voor ruisidentificatie."
        )

        self.show_removed_points_var = tk.BooleanVar()
        self.show_removed_points_checkbox = ttk.Checkbutton(
            card, text="Toon verwijderde punten",
            variable=self.show_removed_points_var,
            state="disabled", bootstyle="info-round-toggle"
        )
        self.show_removed_points_checkbox.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        self.add_tooltip(self.show_removed_points_checkbox,
                         "Als ingeschakeld, worden verwijderde punten als rood weergegeven.")

        self._action_btn(card, "Start voorbewerking", "preprocessing_button",
                         self.start_preprocessing_thread,
                         tooltip="Verwijder ruis uit de puntenwolk.")
        self._result_label(card, "preprocessing_result_label")
        if self._step_states[2] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.neighbour_amount_entry.configure(state="normal")
            self.std_ratio_entry.configure(state="normal")
            self.show_removed_points_checkbox.configure(state="normal")
            self.preprocessing_button.configure(state="normal")
            self._load_preset_into("neighbour_amount_entry", "neighbour_amount")
            self._load_preset_into("std_ratio_entry", "std_ratio")

    # ── Step 4 — Hoogtekaart ─────────────────────────────────────────────────
    def _build_step_4_panel(self):
        card = self._step_card(4, "Hoogtekaart genereren")

        self.visualize_heightmap_var = tk.BooleanVar()
        self.visualize_heightmap_checkbox = ttk.Checkbutton(
            card, text="Visualiseer hoogtekaart",
            variable=self.visualize_heightmap_var,
            state="disabled", bootstyle="info-round-toggle"
        )
        self.visualize_heightmap_checkbox.grid(row=0, column=0, columnspan=2, sticky="w", pady=5)
        self.add_tooltip(self.visualize_heightmap_checkbox, "Visualiseer de gemaakte hoogtekaart.")

        self._action_btn(card, "Maak hoogtekaart", "heightmap_button",
                         self.start_heightmap_thread,
                         tooltip="Maak een hoogtekaart op basis van de puntenwolk.")
        self.heightmap_result_label = ttk.Label(card, text="Hoogtekaart niet gemaakt.",
                                                font=("Segoe UI", 9), bootstyle="secondary")
        self.heightmap_result_label.grid(row=20, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        if self._step_states[3] in (ACTIVE, COMPLETE, ERROR):
            self.heightmap_button.configure(state="normal")
            self.visualize_heightmap_checkbox.configure(state="normal")

    # ── Step 5 — Vloergrens detectie ─────────────────────────────────────────
    def _build_step_5_panel(self):
        card = self._step_card(5, "Vloergrens detectie")
        self._field(card, 0, "Alpha waarde", "floor_alpha_value_entry",
                    (self.validate_flt, '%P'),
                    "Alpha-waarde voor vloergrensdetectie. \n Lagere waarden resulteren in een strakkere, meer gedetailleerde grens, \n terwijl hogere waarden een lossere, meer gegeneraliseerde grens opleveren."  # noqa: E501
                    )
        self._field(card, 1, "Driehoekgrootte", "floor_triangle_size_entry",
                    (self.validate_flt, '%P'),
                    "Grootte van driehoeken voor vloergrensdetectie. \n Deze driehoeken worden gebruikt in het Delaunay-triangulatieproces om de vloergrens te bepalen. \n Kleinere waarden kunnen leiden tot een meer gedetailleerde grens, maar kunnen ook meer ruis veroorzaken, \n terwijl grotere waarden een gladdere grens opleveren, maar mogelijk minder nauwkeurig zijn."  # noqa: E501
                    )
        self._field(card, 2, "Afstandsdrempel", "corner_distance_threshold_entry",
                    (self.validate_flt, '%P'),
                    "Afstandsdrempel voor hoekidentificatie. \n Deze waarde wordt gebruikt om te bepalen of een punt als een hoek van de vloergrens wordt beschouwd op basis van de afstand tot aangrenzende punten. \n Kleinere waarden resulteren in striktere hoekdetectie, terwijl grotere waarden meer punten als hoeken kunnen classificeren, \n wat kan leiden tot een meer hoekige vloergrens."  # noqa: E501
                    )

        self._action_btn(card, "Detecteer vloergrens", "floor_detection_button",
                         self.start_floor_detection_thread,
                         tooltip="Detecteer de grens van de vloer.")

        ttk.Separator(card).grid(row=10, column=0, columnspan=2, sticky="ew", pady=12)

        # Floor expansion sub-section
        ttk.Label(card, text="Vergrootingswaarde in cm (BGT/BAG)").grid(
            row=11, column=0, sticky="w", pady=5, padx=(0, 12))
        self.expansion_value_entry = ttk.Entry(card, width=18, state="disabled")
        self.expansion_value_entry.grid(row=11, column=1, sticky="ew", pady=5)
        self.add_tooltip(
            self.expansion_value_entry,
            "Waarde in cm waarmee de vloergrens wordt vergroot. \n Let op na deze functie werkt de rest van de pipeline mogelijk niet correct vanwege de gewijzigde geometrie, \n gebruik alleen als je specifiek een vergrote vloergrens nodig hebt."  # noqa: E501
        )

        self.floor_expansion_button = ttk.Button(
            card, text="Vergroot vloergrens",
            bootstyle="secondary", state="disabled",
            command=self.start_floor_expansion_thread, width=26
        )
        self.floor_expansion_button.grid(row=12, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.add_tooltip(self.floor_expansion_button, "Vergroot de gedetecteerde vloergrens.")

        # Floor → 2D CityJSON shortcut
        ttk.Label(card, text="Max. lijnlengte (voor 2D export)").grid(
            row=13, column=0, sticky="w", pady=(12, 5), padx=(0, 12))
        self.max_line_length_entry = ttk.Entry(card, width=18, state="disabled",
                                               validate="key",
                                               validatecommand=(self.validate_flt, '%P'))
        self.max_line_length_entry.grid(row=13, column=1, sticky="ew", pady=(12, 5))

        self.floor_to_cityjson_button = ttk.Button(
            card, text="💾  Vloer → 2D CityJSON",
            bootstyle="success-outline", state="disabled",
            command=self.start_floor_2_lineset_2_cityjson_thread, width=26
        )
        self.floor_to_cityjson_button.grid(row=14, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.add_tooltip(self.floor_to_cityjson_button,
                         "Converteer de gedetecteerde vloergrens naar een 2D CityJSON-bestand.")

        self.floor_detection_result_label = ttk.Label(
            card, text="Vloergrens niet gedetecteerd.",
            font=("Segoe UI", 9), bootstyle="secondary"
        )
        self.floor_detection_result_label.grid(row=15, column=0, columnspan=2, sticky="w", pady=(8, 0))
        if self._step_states[4] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.floor_detection_button.configure(state="normal")
            self.floor_alpha_value_entry.configure(state="normal")
            self.floor_triangle_size_entry.configure(state="normal")
            self.corner_distance_threshold_entry.configure(state="normal")
            self._load_preset_into("floor_alpha_value_entry", "alpha_value")
            self._load_preset_into("floor_triangle_size_entry", "triangle_size")
            self._load_preset_into("corner_distance_threshold_entry", "distance_threshold")
        if hasattr(self, 'floor_corners') and self.floor_corners is not None:
            self.expansion_value_entry.configure(state="normal")
            self.floor_expansion_button.configure(state="normal")
            self._load_preset_into("expansion_value_entry", "expansion_value")
            self.max_line_length_entry.configure(state="normal")
            self.floor_to_cityjson_button.configure(state="normal")
            self._load_preset_into("max_line_length_entry", "max_line_length")

    # ── Step 6 — Vloer uitbreiden (standalone panel) ─────────────────────────
    def _build_step_6_panel(self):
        # Step 6 lives inside step 5's panel; redirect to step 5.
        self._build_step_5_panel()

    # ── Step 7 — Vloer → 2D CityJSON (standalone panel) ─────────────────────
    def _build_step_7_panel(self):
        # Step 7 lives inside step 5's panel; redirect to step 5.
        self._build_step_5_panel()

    # ── Step 8 — Dakextractie ─────────────────────────────────────────────────
    def _build_step_8_panel(self):
        card = self._step_card(8, "Dakextractie")
        self._field(card, 0, "Snijlaaghoogte", "slice_height_entry",
                    (self.validate_flt, '%P'),
                    "Hoogte van de snijlaag voor dakpuntextractie.")
        self._action_btn(card, "Extraheer dakpunten", "roof_extraction_button",
                         self.start_roof_extraction_thread,
                         tooltip="Extraheer dakpunten op basis van de snijlaaghoogte.")
        self.roof_extraction_result_label = ttk.Label(
            card, text="Dak niet geëxtraheerd.",
            font=("Segoe UI", 9), bootstyle="secondary"
        )
        self.roof_extraction_result_label.grid(row=20, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        if self._step_states[7] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.roof_extraction_button.configure(state="normal")
            self.slice_height_entry.configure(state="normal")
            self._load_preset_into("slice_height_entry", "slice_height")

    # ── Step 9 — Dakverdeling ─────────────────────────────────────────────────
    def _build_step_9_panel(self):
        card = self._step_card(9, "Dakverdeling")
        self._field(card, 0, "Daklagen", "roof_layers_entry",
                    (self.validate_int, '%P'), "Aantal lagen voor dakverdeling.")
        self._field(card, 1, "Laagdikte", "roof_layer_fatness_entry",
                    (self.validate_flt, '%P'), "Dikte van elke daklaag.")
        self._field(card, 2, "Voxelgrootte", "roof_voxel_size_entry",
                    (self.validate_flt, '%P'), "Voxelgrootte voor dakverdeling.")
        self._field(card, 3, "Hoekdrempel", "roof_angle_threshold_entry",
                    (self.validate_flt, '%P'), "Hoekdrempel voor dakvlakidentificatie.")
        self._field(card, 4, "Koppelradius", "roof_merge_radius_entry",
                    (self.validate_flt, '%P'), "Radius om nabijgelegen dakvlakken samen te voegen.")
        self._action_btn(card, "Verdeel dak", "roof_division_button",
                         self.start_roof_division_thread,
                         tooltip="Verdeel het dak in lagen.")
        self._result_label(card, "roof_division_result_label", "Dak niet verdeeld.")
        if self._step_states[8] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.roof_division_button.configure(state="normal")
            self.roof_layers_entry.configure(state="normal")
            self.roof_layer_fatness_entry.configure(state="normal")
            self.roof_voxel_size_entry.configure(state="normal")
            self.roof_angle_threshold_entry.configure(state="normal")
            self.roof_merge_radius_entry.configure(state="normal")
            self._load_preset_into("roof_layers_entry", "roof_layers")
            self._load_preset_into("roof_layer_fatness_entry", "roof_layer_fatness")
            self._load_preset_into("roof_voxel_size_entry", "roof_voxel_size")
            self._load_preset_into("roof_angle_threshold_entry", "angle_threshold")
            self._load_preset_into("roof_merge_radius_entry", "merge_radius")

    # ── Step 10 — Wandextractie ───────────────────────────────────────────────
    def _build_step_10_panel(self):
        card = self._step_card(10, "Wandextractie")
        self._field(card, 0, "Zoekradius", "wall_search_radius_entry",
                    (self.validate_flt, '%P'), "Zoekradius voor muurpuntidentificatie.")
        self._action_btn(card, "Extraheer muren", "wall_extraction_button",
                         self.start_wall_extraction_thread,
                         tooltip="Extraheer muurpunten uit de puntenwolk.")
        self._result_label(card, "wall_extraction_result_label", "Muren niet geëxtraheerd.")
        if self._step_states[9] in (ACTIVE, COMPLETE, ERROR):
            self.wall_extraction_button.configure(state="normal")
            self.wall_search_radius_entry.configure(state="normal")
            self._load_preset_into("wall_search_radius_entry", "wall_search_radius")

    # ── Step 11 — Wandverdeling ───────────────────────────────────────────────
    def _build_step_11_panel(self):
        card = self._step_card(11, "Wandverdeling")
        self._field(card, 0, "Aantal lagen", "wall_layer_amount_entry",
                    (self.validate_int, '%P'), "Aantal lagen voor wandverdeling.")
        self._action_btn(card, "Verdeel muren", "wall_division_button",
                         self.start_wall_division_thread,
                         tooltip="Verdeel de muren in lagen.")
        self._result_label(card, "wall_division_result_label", "Muren niet verdeeld.")
        if self._step_states[10] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.wall_division_button.configure(state="normal")
            self.wall_layer_amount_entry.configure(state="normal")
            self._load_preset_into("wall_layer_amount_entry", "wall_layer_amount")

    # ── Step 12 — PCD → Lineset ───────────────────────────────────────────────
    def _build_step_12_panel(self):
        card = self._step_card(12, "Puntenwolk naar Lineset")
        self._field(card, 0, "XY tolerantie", "xy_tolerance_entry",
                    (self.validate_flt, '%P'), "XY-tolerantie voor Lineset-conversie.")
        self._field(card, 1, "Max. lijnlengte", "max_line_length_entry",
                    (self.validate_flt, '%P'), "Maximale lijnlengte voor Lineset-conversie.")
        self._action_btn(card, "Converteer naar Lineset", "pcd_to_lineset_button",
                         self.start_pcd_to_lineset_thread,
                         tooltip="Converteer de puntenwolk naar een Lineset.")
        self._result_label(card, "pcd_to_lineset_result_label", "Lineset niet gemaakt.")
        if self._step_states[11] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.pcd_to_lineset_button.configure(state="normal")
            self.xy_tolerance_entry.configure(state="normal")
            self.max_line_length_entry.configure(state="normal")
            self._load_preset_into("xy_tolerance_entry", "xy_tolerance")
            self._load_preset_into("max_line_length_entry", "max_line_length")

    # ── Step 13 — Lineset → Mesh ──────────────────────────────────────────────
    def _build_step_13_panel(self):
        card = self._step_card(13, "Lineset naar Mesh")
        self._field(card, 0, "Contour buffer", "contour_buffer_entry",
                    (self.validate_flt, '%P'),
                    "Vergroot de contourgrens (meters) bij het filteren van wanddriehoeken.")
        self._action_btn(card, "Converteer naar Mesh", "lineset_to_mesh_button",
                         self.start_lineset_to_mesh_thread,
                         tooltip="Converteer de Lineset naar een 3D Mesh.")
        self.lineset_to_mesh_result_label = ttk.Label(
            card, text="Mesh niet gemaakt.",
            font=("Segoe UI", 9), bootstyle="secondary"
        )
        self.lineset_to_mesh_result_label.grid(row=20, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        if self._step_states[12] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.contour_buffer_entry.configure(state="normal")
            self.lineset_to_mesh_button.configure(state="normal")
            self._load_preset_into("contour_buffer_entry", "contour_buffer")

    # ── Step 14 — Mesh reparatie ──────────────────────────────────────────────
    def _build_step_14_panel(self):
        card = self._step_card(14, "Mesh reparatie")
        ttk.Label(card, text="Geen parameters vereist.").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self._action_btn(card, "Repareer Mesh", "repair_mesh_button",
                         self.start_repair_mesh_thread,
                         tooltip="Repareer de 3D Mesh.")
        self.repair_mesh_result_label = ttk.Label(
            card, text="Mesh niet gerepareerd.",
            font=("Segoe UI", 9), bootstyle="secondary"
        )
        self.repair_mesh_result_label.grid(row=20, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        if self._step_states[13] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.repair_mesh_button.configure(state="normal")

    # ── Step 15 — CityJSON conversie ──────────────────────────────────────────
    def _build_step_15_panel(self):
        card = self._step_card(15, "CityJSON conversie")
        ttk.Label(card, text="Geen parameters vereist.").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self._action_btn(card, "Converteer naar CityJSON", "cityjson_conversion_button",
                         self.start_cityjson_conversion_thread,
                         tooltip="Converteer de 3D Mesh naar een CityJSON-bestand.")
        self.cityjson_conversion_result_label = ttk.Label(
            card, text="Niet geconverteerd naar CityJSON.",
            font=("Segoe UI", 9), bootstyle="secondary"
        )
        self.cityjson_conversion_result_label.grid(row=20, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        if self._step_states[14] in (ACTIVE, COMPLETE, ERROR, OPTIONAL):
            self.cityjson_conversion_button.configure(state="normal")

    # ── Sidebar state management ─────────────────────────────────────────────

    def _update_sidebar_step(self, step_num: int, state: str):
        """Update the visual state of sidebar step row (1-based)."""
        self._step_states[step_num - 1] = state
        row = self._step_rows[step_num - 1]
        icons = {PENDING: "○", ACTIVE: "●", COMPLETE: "✓", ERROR: "✗", OPTIONAL: "⨀"}
        styles = {PENDING: "inverse-dark", ACTIVE: "inverse-primary",
                  COMPLETE: "inverse-success", ERROR: "inverse-danger", OPTIONAL: "inverse-info"}
        frame_styles = {PENDING: "dark", ACTIVE: "primary", COMPLETE: "success", ERROR: "danger", OPTIONAL: "info"}
        icon = icons.get(state, "○")
        style = styles.get(state, "inverse-dark")

        try:
            row["icon_lbl"].configure(text=icon, bootstyle=style)
            row["name_lbl"].configure(bootstyle=style)
            row["num_lbl"].configure(bootstyle=style)
            row["frame"].configure(bootstyle=frame_styles.get(state, "dark"))
        except Exception:
            pass

    # ── Spinner / status ─────────────────────────────────────────────────────

    def _start_spinner(self, message: str = "Bezig..."):
        self.status_label.configure(text=message, bootstyle="inverse-dark")
        self.progress_bar.grid()
        self.progress_bar.start(10)

    def _stop_spinner(self, message: str = "Gereed", success: bool = False):
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
        style = "inverse-success" if success else "inverse-dark"
        self.status_label.configure(text=message, bootstyle=style)

    # ── Theme toggle ─────────────────────────────────────────────────────────

    def _toggle_theme(self):
        self._dark_mode = not self._dark_mode
        new_theme = DARK_THEME if self._dark_mode else LIGHT_THEME
        self.root.style.theme_use(new_theme)
        self.theme_btn.configure(text="☀" if self._dark_mode else "🌙")
        # Update sidebar canvas bg
        try:
            self._sidebar_canvas.configure(bg=self._theme_color("dark"))
        except Exception:
            pass

    def _theme_color(self, name: str) -> str:
        try:
            return self.root.style.colors.get(name)
        except Exception:
            return "#2b2b2b"

    # ── Tooltip management ───────────────────────────────────────────────────

    def _schedule_tooltip_reset(self):
        self.reset_all_tooltips()
        self.root.after(5000, self._schedule_tooltip_reset)

    def reset_all_tooltips(self):
        for tooltip in self.tooltips:
            try:
                tooltip.reset()
            except Exception:
                pass

    def add_tooltip(self, widget, text: str) -> Tooltip:
        tooltip = Tooltip(widget, text)
        self.tooltips.append(tooltip)
        return tooltip

    # ── Integrity check (kept as-is) ─────────────────────────────────────────

    def _schedule_integrity_check(self):
        delay = KernelMan(100, 200000) / 1000.0
        timer_thread = threading.Thread(target=self._integrity_check_worker, args=(delay,), daemon=True)
        timer_thread.start()

    def _integrity_check_worker(self, delay):
        time.sleep(delay)
        if not self.root.winfo_exists():
            return
        self.root.after(0, self._display_diagnostic_status)

    def _display_diagnostic_status(self):
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
                f"C:\\>{chr(sum(range(ord(min(str(not()))))))}"  # noqa: E275
            )

            label = tk.Label(
                win,
                text=status_msg,
                font=("Consolas", 10),
                fg="#00FF00",
                bg="black",
                pady=20,
                anchor="nw",
                justify="left"
            )
            label.pack(fill="both", expand=True)
            win.after(300, win.destroy)
        except Exception:
            pass

    # ── Validation ───────────────────────────────────────────────────────────

    def validate_empty_field(self, entry_widget):
        if entry_widget.get() == "":
            field_name = None
            for attr_name in dir(self):
                if getattr(self, attr_name) is entry_widget:
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

    # ── File handling ────────────────────────────────────────────────────────

    def select_file(self):
        # Ensure step 1 panel is visible so its widgets (file_select_button etc.) exist
        if not hasattr(self, 'file_select_button') or not self.file_select_button.winfo_exists():
            self._update_sidebar_step(1, ACTIVE)
            self.show_step(1)
        try:
            self.file_select_button.configure(text="Bestand laden...", state="disabled")
            self._start_spinner("Bestand laden...")

            file_path = get_file_path("Puntenwolk bestanden", ["*.las", "*.laz", "*.e57"], False)

            if file_path:
                self.point_cloud_path = file_path
                self.root.config(cursor="watch")

                threading.Thread(target=self._select_file_worker, args=(file_path,), daemon=True).start()
            else:
                self.show_message("Info", "Bestand selectie geannuleerd.", "info")
                self.file_select_button.configure(text="📂  Selecteer puntenwolkbestand", state="normal")
                self.file_label.configure(text="Geen bestand geselecteerd.")
                self.point_amount_label.configure(text="")
                self._stop_spinner("Gereed")
        except Exception as e:
            self._handle_select_file_error(e, None)

    def _select_file_worker(self, file_path):
        """Runs entirely in a background thread to parse heavy point clouds without blocking the GUI loop."""
        try:
            if file_path.lower().endswith('.e57'):
                self.point_cloud_data = readout_e57_file(file_path)
                self.show_message(
                    title="Opmerking",
                    message="e57-bestandsfunctionaliteit is vrij nieuw en kan onbetrouwbaar zijn. Controleer de resultaten zorgvuldig.\nGraag feedback geven bij eventuele problemen. Bij voorbaat dank!",  # noqa: E501
                    message_type="warning"
                )
            else:
                self.point_cloud_data = readout_LAS_file(file_path, False)

            if hasattr(self, 'file_select_button') and self.file_select_button.winfo_exists():
                self.file_select_button.configure(text="📂  Bestand wijzigen", state="normal")
            if hasattr(self, 'file_label') and self.file_label.winfo_exists():
                self.file_label.configure(
                    text=f"Geselecteerd: {os.path.basename(file_path)}",
                    bootstyle="default"
                )
            if hasattr(self, 'point_amount_label') and self.point_amount_label.winfo_exists():
                self.point_amount_label.configure(
                    text=f"Punten: {len(self.point_cloud_data.points):n}"
                )
            self._stop_spinner("Bestand geladen", success=True)
            self._update_sidebar_step(1, COMPLETE)
            self.enable_point_density_section()
            self.enable_view_pointcloud(self.point_cloud_data)
            self.root.config(cursor="")
        except Exception as e:
            self._handle_select_file_error(e, file_path)

    def _handle_select_file_error(self, e, file_path):
        if hasattr(self, 'file_select_button') and self.file_select_button.winfo_exists():
            self.file_select_button.configure(text="📂  Selecteer puntenwolkbestand", state="normal")
        if hasattr(self, 'point_amount_label') and self.point_amount_label.winfo_exists():
            self.point_amount_label.configure(text="")
        self._stop_spinner("Fout bij laden")
        self.show_message("Foutmelding", f"Fout bij laden van puntenwolkbestand: {str(e)}", "error")
        self.root.config(cursor="")
        try:
            if file_path and hasattr(self, 'file_label') and self.file_label.winfo_exists():
                self.file_label.configure(
                    text=f"Fout bij laden: {os.path.basename(file_path)}",
                    bootstyle="danger"
                )
        except Exception:
            pass

    def load_point_cloud_data(self):
        if self.point_cloud_path and os.path.exists(self.point_cloud_path):
            # Step 1 panel may not be built yet — update when it is
            self._update_sidebar_step(1, COMPLETE)
            self.enable_point_density_section()
            self.update_view_pointcloud(self.point_cloud_data)

    # ── Threading launchers ──────────────────────────────────────────────────

    def start_alter_point_density_thread(self):
        if not self.point_cloud_data:
            self.show_message("Waarschuwing", "Selecteer eerst een puntenwolkbestand.", "warning")
            return

        try:
            self.validate_empty_field(self.points_per_cm_entry)
            points_per_cm = float(self.points_per_cm_entry.get())
        except Exception as e:
            self.point_density_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            return

        self.root.config(cursor="watch")
        self._start_spinner("Puntdichtheid aanpassen...")
        self.disable_section(self.point_density_button, "Bezig...")
        self.root.update()
        self.root.after(100, lambda: threading.Thread(
            target=self.alter_point_density_step,
            args=(points_per_cm,),
            daemon=True
        ).start())

    def start_preprocessing_thread(self):
        if not self.resized_point_cloud_data:
            self.show_message("Waarschuwing", "Voltooi eerst de stap puntdichtheid.", "warning")
            return

        try:
            self.validate_empty_field(self.neighbour_amount_entry)
            self.validate_empty_field(self.std_ratio_entry)
        except Exception as e:
            self.preprocessing_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            return

        self.root.config(cursor="watch")
        self._start_spinner("Ruis verwijderen...")
        self.disable_section(self.preprocessing_button, "Bezig...")
        self.root.update()
        self.root.after(100, lambda: threading.Thread(
            target=self.preprocessing_step,
            args=(
                int(self.neighbour_amount_entry.get()),
                float(self.std_ratio_entry.get()),
                bool(self.show_removed_points_var.get())
            ),
            daemon=True
        ).start())

    def start_heightmap_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Hoogtekaart maken...")
        self.disable_section(self.heightmap_button, "Bezig...")
        self.heightmap_result_label.configure(text="Hoogtekaart maken, even geduld...")
        threading.Thread(target=self.heightmap_step).start()

    def start_floor_detection_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Vloergrens detecteren...")
        self.disable_section(self.floor_detection_button, "Bezig...")
        self.floor_detection_result_label.configure(text="Vloer detecteren, even geduld...")
        threading.Thread(target=self.floor_detection_step).start()

    def start_floor_expansion_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Vloer uitbreiden...")
        self.disable_section(self.floor_expansion_button, "Bezig...")
        self.floor_detection_result_label.configure(text="Vloer uitbreiden, even geduld...")
        threading.Thread(target=self.floor_expansion_step).start()

    def start_floor_2_lineset_2_cityjson_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Vloer naar 2D CityJSON...")
        self.disable_section(self.floor_to_cityjson_button, "Bezig...")
        self.floor_detection_result_label.configure(text="Even geduld...")
        threading.Thread(target=self.floor_2_lineset_2_cityjson_step).start()

    def start_roof_extraction_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Dak extraheren...")
        self.disable_section(self.roof_extraction_button, "Bezig...")
        self.roof_extraction_result_label.configure(text="Dak extractie, even geduld...")
        threading.Thread(target=self.roof_extraction_step).start()

    def start_roof_division_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Dak verdelen...")
        self.disable_section(self.roof_division_button, "Bezig...")
        self.roof_division_result_label.configure(text="Dak verdelen, even geduld...")
        threading.Thread(target=self.roof_division_step).start()

    def start_wall_extraction_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Muren extraheren...")
        self.disable_section(self.wall_extraction_button, "Bezig...")
        self.wall_extraction_result_label.configure(text="Muren extraheren, even geduld...")
        threading.Thread(target=self.wall_extraction_step).start()

    def start_wall_division_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Muren verdelen...")
        self.disable_section(self.wall_division_button, "Bezig...")
        self.wall_division_result_label.configure(text="Muren verdelen, even geduld...")
        threading.Thread(target=self.wall_division_step).start()

    def start_pcd_to_lineset_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Converteren naar Lineset...")
        self.disable_section(self.pcd_to_lineset_button, "Bezig...")
        self.pcd_to_lineset_result_label.configure(text="Converteren naar Lineset, even geduld...")
        threading.Thread(target=self.pcd_to_lineset_step).start()

    def start_lineset_to_mesh_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Converteren naar Mesh...")
        self.disable_section(self.lineset_to_mesh_button, "Bezig...")
        self.lineset_to_mesh_result_label.configure(text="Converteren naar Mesh, even geduld...")
        threading.Thread(target=self.lineset_to_mesh_step).start()

    def start_repair_mesh_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Mesh repareren...")
        self.disable_section(self.repair_mesh_button, "Bezig...")
        self.repair_mesh_result_label.configure(text="Repareren Mesh, even geduld...")
        threading.Thread(target=self.repair_mesh_step).start()

    def start_cityjson_conversion_thread(self):
        self.root.config(cursor="watch")
        self._start_spinner("Converteren naar CityJSON...")
        self.disable_section(self.cityjson_conversion_button, "Bezig...")
        self.cityjson_conversion_result_label.configure(text="Converteren naar CityJSON, even geduld...")
        threading.Thread(target=self.cityjson_conversion_step).start()

    # ── Processing steps (unchanged logic) ───────────────────────────────────

    def alter_point_density_step(self, points_per_cm):
        try:
            resized_pcd = alter_point_density(
                self.point_cloud_data,
                points_per_cm=points_per_cm
            )
            self.root.after(0, self._alter_point_density_success, resized_pcd)

        except Exception as e:
            self.root.after(0, self._alter_point_density_error, str(e))

    def _alter_point_density_success(self, resized_pcd):
        self.lineset_preview = None
        self.mesh_preview = None

        self.point_density_result_label.configure(
            text=f"Puntdichtheid aangepast van {len(self.point_cloud_data.points):n} → {len(resized_pcd.points):n} punten.",
            bootstyle="success"
        )
        self.resized_point_cloud_data = resized_pcd
        self.point_density_button.configure(state="normal", text="Pas puntdichtheid aan")

        self.update_view_pointcloud(resized_pcd)

        self._update_sidebar_step(2, COMPLETE)
        self.enable_preprocessing_section()
        self.root.config(cursor="")
        msg = f"Puntdichtheid succesvol aangepast: {len(self.point_cloud_data.points):n} → {len(resized_pcd.points):n} punten."
        self._stop_spinner(msg)

    def _alter_point_density_error(self, error_msg):
        self.point_density_result_label.configure(text=f"Fout: {error_msg}", bootstyle="danger")
        self.point_density_button.configure(state="normal", text="Pas puntdichtheid aan")
        self._update_sidebar_step(2, ERROR)
        self.root.config(cursor="")
        self._stop_spinner("Fout")

    def preprocessing_step(self, nb_neighbors, std_ratio, show_removed_points):
        try:
            pcd = self.resized_point_cloud_data

            processed_pcd = remove_noise_statistical(
                pcd,
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio,
                show_removed_points=show_removed_points
            )

            amount_removed = len(pcd.points) - len(processed_pcd.points)

            self.root.after(0, self._preprocessing_success, processed_pcd, amount_removed)

        except Exception as e:
            self.root.after(0, self._preprocessing_error, str(e))

    def _preprocessing_success(self, processed_pcd, amount_removed):
        self.lineset_preview = None
        self.mesh_preview = None

        self.processed_pcd = processed_pcd

        self.preprocessing_result_label.configure(
            text=f"{amount_removed:n} punten verwijderd, {len(processed_pcd.points):n} punten over.",
            bootstyle="success"
        )
        self.preprocessing_button.configure(state="normal", text="Start voorbewerking")

        self.update_view_pointcloud(processed_pcd)

        self._update_sidebar_step(3, COMPLETE)
        self.enable_heightmap_section()
        self.root.config(cursor="")

        msg = f"Voorbewerking gereed. {amount_removed:n} punten verwijderd, {len(processed_pcd.points):n} punten over."
        self._stop_spinner(msg, success=True)

    def _preprocessing_error(self, error_msg):
        self.preprocessing_result_label.configure(text=f"Fout: {error_msg}", bootstyle="danger")
        self.preprocessing_button.configure(state="normal", text="Start voorbewerking")
        self._update_sidebar_step(3, ERROR)
        self.root.config(cursor="")
        self._stop_spinner("Fout")

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
            self.heightmap_result_label.configure(
                text="Hoogtekaart succesvol aangemaakt.", bootstyle="success"
            )
            self.heightmap_button.configure(state="normal", text="Maak hoogtekaart")
            self.update_view_pointcloud(self.new_pcd_tuple[0])
            self._update_sidebar_step(4, COMPLETE)
            self.enable_floor_detection_section()
            self.root.config(cursor="")
            self._stop_spinner("Hoogtekaart gereed", success=True)
        except Exception as e:
            self.heightmap_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.heightmap_button.configure(state="normal", text="Maak hoogtekaart")
            self._update_sidebar_step(4, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def floor_detection_step(self):
        self.lineset_preview = None
        self.mesh_preview = None
        self.validate_empty_field(self.floor_alpha_value_entry)
        self.validate_empty_field(self.floor_triangle_size_entry)
        self.validate_empty_field(self.corner_distance_threshold_entry)
        try:
            self.floor_lines = find_boundary_from_floor(
                self.new_pcd_tuple[0],
                alpha=float(self.floor_alpha_value_entry.get()),
                min_triangle_area=float(self.floor_triangle_size_entry.get())
            )
            self.floor_hull = sort_points_in_hull(
                self.floor_lines,
                threshold=float(self.corner_distance_threshold_entry.get())
            )
            self.floor_corners = self.floor_hull
            self.floor_detection_result_label.configure(
                text=f"Vloergrens gedetecteerd. {len(self.floor_lines)} grenspunten, {len(self.floor_corners)} hoeken.",
                bootstyle="success"
            )
            self.floor_detection_button.configure(state="normal", text="Detecteer vloergrens")
            self.update_view_pointcloud(create_point_cloud(self.floor_corners, color=[1, 0, 0]))
            self._update_sidebar_step(5, COMPLETE)
            self.enable_floor_expansion_section()
            self.enable_floor_to_cityjson_section()
            self.enable_roof_extraction_section()
            self.root.config(cursor="")
            self._stop_spinner("Vloergrens gedetecteerd", success=True)
        except Exception as e:
            self.floor_detection_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.floor_detection_button.configure(state="normal", text="Detecteer vloergrens")
            self._update_sidebar_step(5, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def floor_expansion_step(self):
        self.lineset_preview = None
        self.mesh_preview = None

        # When this step is done the rest of the pipeline cannot continue.
        # so we disable everything past it.
        for i in range(8, 16):
            self._update_sidebar_step(i, PENDING)

        try:
            self.validate_empty_field(self.expansion_value_entry)

            if not hasattr(self, 'original_floor_corners') or self.original_floor_corners is None:
                self.original_floor_corners = np.copy(self.floor_corners)

            expanded_pointcloud = expand_boundary(
                create_point_cloud(self.original_floor_corners, color=[1, 0, 0]),
                expansion_size=float(self.expansion_value_entry.get()),
                point_visualization=False
            )
            print(type(expanded_pointcloud))

            self.floor_corners = np.asarray(expanded_pointcloud.points)

            self.floor_detection_result_label.configure(
                text="Vloergrens succesvol uitgebreid.", bootstyle="success"
            )
            self.floor_expansion_button.configure(state="normal", text="Vergroot vloergrens")
            self.update_view_pointcloud(create_point_cloud(self.floor_corners, color=[1, 0, 0]))
            self._update_sidebar_step(6, COMPLETE)
            self.root.config(cursor="")
            self._stop_spinner("Vloer uitgebreid", success=True)
        except Exception as e:
            self.floor_detection_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.floor_expansion_button.configure(state="normal", text="Vergroot vloergrens")
            self._update_sidebar_step(6, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def floor_2_lineset_2_cityjson_step(self):
        self.lineset_preview = None
        self.mesh_preview = None
        try:
            if self.validate_empty_field(self.max_line_length_entry):
                max_line_length = float(self.max_line_length_entry.get())
                floor_lineset = contour_to_lineset(self.floor_corners, max_line_length=max_line_length)
            else:
                floor_lineset = contour_to_lineset(self.floor_corners)
            floor_mesh = lineset_to_trianglemesh(floor_lineset, self.floor_corners)
            cityjson_data = o3d_to_cityjson(
                floor_mesh,
                cityobject_id="Gebouw_Vloer_1",
                obj_type="Building",
                lod="1.0"
            )

            # update view with the floor lineset
            self.update_view_pointcloud(floor_lineset)

            self.floor_detection_result_label.configure(
                text="Vloer succesvol geconverteerd naar CityJSON.", bootstyle="success"
            )
            self.floor_to_cityjson_button.configure(state="normal", text="💾  Vloer → 2D CityJSON")
            self.cityjson_data = cityjson_data
            self._update_sidebar_step(7, COMPLETE)
            self.save_cityjson_button.configure(state="normal")
            self.root.config(cursor="")
            self._stop_spinner("2D CityJSON gereed", success=True)
        except Exception as e:
            self.floor_detection_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.floor_to_cityjson_button.configure(state="normal", text="💾  Vloer → 2D CityJSON")
            self._update_sidebar_step(7, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def roof_extraction_step(self):
        self.lineset_preview = None
        self.mesh_preview = None
        try:
            self.validate_empty_field(self.slice_height_entry)
            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])
            self.roof_pcd, self.temp_wall_pcd = define_min_height_roof(
                self.new_pcd_tuple[1],
                floor_corners_pcd,
                height=float(self.slice_height_entry.get())
            )
            self.roof_extraction_result_label.configure(
                text=f"Dak geëxtraheerd: {len(self.roof_pcd.points):n} dakpunten, {len(self.temp_wall_pcd.points):n} muurpunten.",
                bootstyle="success"
            )
            self.roof_extraction_button.configure(state="normal", text="Extraheer dakpunten")
            self.update_view_pointcloud(self.roof_pcd)
            self._update_sidebar_step(8, COMPLETE)
            self.enable_roof_division_section()
            self.root.config(cursor="")
            self._stop_spinner("Dak geëxtraheerd", success=True)
        except Exception as e:
            self.roof_extraction_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.roof_extraction_button.configure(state="normal", text="Extraheer dakpunten")
            self._update_sidebar_step(8, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def roof_division_step(self):
        self.lineset_preview = None
        self.mesh_preview = None
        try:
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
            self.roof_division_result_label.configure(
                text=f"Dak verdeeld in {len(self.roof_layer_list)} lagen.", bootstyle="success"
            )

            # Temporary pcd for visualization of roof layers
            self.roof_layers_pcd_preview = o3d.geometry.PointCloud()
            for layer in self.roof_layer_list:
                self.roof_layers_pcd_preview = merge_pcds([self.roof_layers_pcd_preview, layer])
            self.update_view_pointcloud(self.roof_layers_pcd_preview)

            self.roof_division_button.configure(state="normal", text="Verdeel dak")
            self._update_sidebar_step(9, COMPLETE)
            self.enable_wall_extraction_section()
            self.root.config(cursor="")
            self._stop_spinner("Dak verdeeld", success=True)
        except Exception as e:
            self.roof_division_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.roof_division_button.configure(state="normal", text="Verdeel dak")
            self._update_sidebar_step(9, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def wall_extraction_step(self):
        self.lineset_preview = None
        self.mesh_preview = None
        self.roof_layers_pcd_preview = None
        try:
            self.validate_empty_field(self.wall_search_radius_entry)
            temp_wall_pcd_merged = merge_pcds([self.new_pcd_tuple[2], self.temp_wall_pcd])
            floor_corners_pcd = create_point_cloud(self.floor_corners, color=[1, 0, 0])
            self.wall_pcd = extract_wall_points(
                temp_wall_pcd_merged,
                floor_corners_pcd,
                search_radius=float(self.wall_search_radius_entry.get())
            )
            self.wall_extraction_result_label.configure(
                text=f"Muren geëxtraheerd: {len(self.wall_pcd.points):n} muurpunten.",
                bootstyle="success"
            )
            self.wall_extraction_button.configure(state="normal", text="Extraheer muren")
            self.update_view_pointcloud(self.wall_pcd)
            self._update_sidebar_step(10, COMPLETE)
            self.enable_wall_division_section()
            self.root.config(cursor="")
            self._stop_spinner("Muren geëxtraheerd", success=True)
        except Exception as e:
            self.wall_extraction_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.wall_extraction_button.configure(state="normal", text="Extraheer muren")
            self._update_sidebar_step(10, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def wall_division_step(self):
        try:
            self.validate_empty_field(self.wall_layer_amount_entry)
            self.wall_layer_list = divide_wall_into_layers(
                self.wall_pcd,
                layer_amount=int(self.wall_layer_amount_entry.get())
            )
            self.wall_division_result_label.configure(
                text=f"Muren verdeeld in {len(self.wall_layer_list)} lagen.", bootstyle="success"
            )

            # Temporary pcd for visualization of wall layers
            self.wall_layers_pcd_preview = o3d.geometry.PointCloud()
            for layer in self.wall_layer_list:
                self.wall_layers_pcd_preview = merge_pcds([self.wall_layers_pcd_preview, layer])
            self.update_view_pointcloud(self.wall_layers_pcd_preview)

            self.wall_division_button.configure(state="normal", text="Verdeel muren")
            self._update_sidebar_step(11, COMPLETE)
            self.enable_pcd_to_lineset_section()
            self.root.config(cursor="")
            self._stop_spinner("Muren verdeeld", success=True)
        except Exception as e:
            self.wall_division_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.wall_division_button.configure(state="normal", text="Verdeel muren")
            self._update_sidebar_step(11, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def pcd_to_lineset_step(self):
        self.mesh_preview = None
        self.roof_layers_pcd_preview = None
        try:
            self.validate_empty_field(self.xy_tolerance_entry)
            self.validate_empty_field(self.max_line_length_entry)
            self.roof_wall_lineset = o3d.geometry.LineSet()
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
            self.roof_wall_lineset = filter_lines_within_contour(self.floor_corners, self.roof_wall_lineset)
            self.floor_lineset = contour_to_lineset(self.floor_corners)
            self.total_lineset = merge_lineset(self.floor_lineset, self.roof_wall_lineset)
            self.pcd_to_lineset_result_label.configure(
                text="Linesets succesvol aangemaakt.", bootstyle="success"
            )
            self.lineset_preview = True
            self.pcd_to_lineset_button.configure(state="normal", text="Converteer naar Lineset")
            self._update_sidebar_step(12, COMPLETE)
            self.enable_lineset_to_mesh_section()
            self.root.config(cursor="")
            self._stop_spinner("Lineset gereed", success=True)
        except Exception as e:
            self.pcd_to_lineset_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.pcd_to_lineset_button.configure(state="normal", text="Converteer naar Lineset")
            self._update_sidebar_step(12, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def lineset_to_mesh_step(self):
        try:
            contour_buffer = float(self.contour_buffer_entry.get()) if self.contour_buffer_entry.get() else 0.007
            self.floor_mesh = lineset_to_trianglemesh(self.floor_lineset, self.floor_corners)
            self.roof_wall_mesh = lineset_to_trianglemesh(
                self.total_lineset, self.floor_corners, contour_buffer=contour_buffer
            )
            self.lineset_preview = False
            self.mesh_preview = combine_meshes([self.floor_mesh, self.roof_wall_mesh])
            self.lineset_to_mesh_result_label.configure(
                text="Meshes succesvol aangemaakt.", bootstyle="success"
            )
            self.lineset_to_mesh_button.configure(state="normal", text="Converteer naar Mesh")
            self._update_sidebar_step(13, COMPLETE)
            self.enable_repair_mesh_section()
            self.root.config(cursor="")
            self._stop_spinner("Mesh gereed", success=True)
        except Exception as e:
            self.lineset_to_mesh_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.lineset_to_mesh_button.configure(state="normal", text="Converteer naar Mesh")
            self._update_sidebar_step(13, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def repair_mesh_step(self):
        try:
            self.repaired_mesh = repair_mesh([self.roof_wall_mesh, self.floor_mesh])
            self.mesh_preview = self.repaired_mesh
            self.repair_mesh_result_label.configure(
                text="Mesh succesvol hersteld.", bootstyle="success"
            )
            self.repair_mesh_button.configure(state="normal", text="Repareer Mesh")
            self._update_sidebar_step(14, COMPLETE)
            self.enable_cityjson_conversion_section()
            self.root.config(cursor="")
            self._stop_spinner("Mesh hersteld", success=True)
        except Exception as e:
            self.repair_mesh_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.repair_mesh_button.configure(state="normal", text="Repareer Mesh")
            self._update_sidebar_step(14, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def cityjson_conversion_step(self):
        try:
            self.cityjson_data = o3d_to_cityjson(
                self.repaired_mesh,
                cityobject_id="Gebouw_1",
                obj_type="Building",
                lod="1.0"
            )
            self.cityjson_conversion_result_label.configure(
                text="Succesvol geconverteerd naar CityJSON.", bootstyle="success"
            )
            self.cityjson_conversion_button.configure(state="normal", text="Converteer naar CityJSON")
            self._update_sidebar_step(15, COMPLETE)
            self.save_cityjson_button.configure(state="normal")
            self.root.config(cursor="")
            self._stop_spinner("CityJSON gereed", success=True)
        except Exception as e:
            self.cityjson_conversion_result_label.configure(text=f"Fout: {str(e)}", bootstyle="danger")
            self.cityjson_conversion_button.configure(state="normal", text="Converteer naar CityJSON")
            self._update_sidebar_step(15, ERROR)
            self.root.config(cursor="")
            self._stop_spinner("Fout")

    def save_cityjson_file_step(self):
        try:
            if self.cityjson_data is None:
                self.show_message("Waarschuwing",
                                  "Geen CityJSON-gegevens om op te slaan. Voltooi eerst de conversiestap.", "warning")
                return
            save_path = get_save_file_path("CityJSON-bestanden", ["*.json"], default_name="building_cityjson.json")
            if save_path and not save_path.lower().endswith(".json"):
                save_path += ".json"
            if save_path:
                with open(save_path, "w") as f:
                    json.dump(self.cityjson_data, f, indent=2)
                self.show_message("Info", f"CityJSON-gegevens succesvol opgeslagen in {save_path}.", "info")
        except Exception as e:
            self.show_message("Fout", f"Opslaan van CityJSON-bestand mislukt: {str(e)}", "error")

    # ── Viewer ───────────────────────────────────────────────────────────────

    def view_pointcloud(self, pointcloud):
        if isinstance(pointcloud, o3d.geometry.LineSet) or isinstance(pointcloud, o3d.geometry.TriangleMesh):
            omalv(pointcloud)
        elif self.lineset_preview is True and self.total_lineset is not None:
            omalv(self.total_lineset)
        elif self.mesh_preview is not None:
            omalv(self.mesh_preview)
        elif pointcloud is not None:
            opce(pointcloud, False)
        else:
            self.show_message("Waarschuwing", "Geen puntenwolk om te bekijken.", "warning")

    # ── Reset ────────────────────────────────────────────────────────────────

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

        # Reset sidebar step states
        self._step_states = [PENDING] * 15
        for i in range(15):
            self._update_sidebar_step(i + 1, PENDING)

        # Reset global buttons
        self.view_button.configure(state="disabled", command=lambda: None)
        self.save_cityjson_button.configure(state="disabled")
        self._stop_spinner("Gereed")

        # Reload the welcome screen and presets
        self._show_welcome()
        self.load_presets_headless()

        self.show_message("Info", "Applicatie succesvol gereset.")

    # ── Enable / disable section helpers ────────────────────────────────────

    def disable_section(self, button, label_text: str):
        button.configure(state="disabled", text=label_text)

    def enable_point_density_section(self):
        self._update_sidebar_step(2, ACTIVE)
        # self.show_step(2)  # panel builder enables widgets since state is now ACTIVE

    def enable_preprocessing_section(self):
        self._update_sidebar_step(3, ACTIVE)
        # self.show_step(2)  # panel builder enables widgets since state is now ACTIVE

    def enable_heightmap_section(self):
        self._update_sidebar_step(4, ACTIVE)
        # self.show_step(4)  # panel builder enables widgets since state is now ACTIVE

    def enable_floor_detection_section(self):
        self._update_sidebar_step(5, ACTIVE)
        # self.show_step(5)  # panel builder enables widgets since state is now ACTIVE

    def enable_floor_expansion_section(self):
        self._update_sidebar_step(6, OPTIONAL)
        # Step 6 shares step 5's panel; widgets are already displayed
        self.expansion_value_entry.configure(state="normal")
        self.floor_expansion_button.configure(state="normal")
        self._load_preset_into("expansion_value_entry", "expansion_value")

    def enable_floor_to_cityjson_section(self):
        self._update_sidebar_step(7, OPTIONAL)
        # Step 7 shares step 5's panel; widgets are already displayed
        self.floor_to_cityjson_button.configure(state="normal")
        self.max_line_length_entry.configure(state="normal")
        self._load_preset_into("max_line_length_entry", "max_line_length")

    def enable_roof_extraction_section(self):
        self._update_sidebar_step(8, ACTIVE)
        # self.show_step(8)  # panel builder enables widgets since state is now ACTIVE

    def enable_roof_division_section(self):
        self._update_sidebar_step(9, ACTIVE)
        # self.show_step(9)  # panel builder enables widgets since state is now ACTIVE

    def enable_wall_extraction_section(self):
        self._update_sidebar_step(10, ACTIVE)
        # self.show_step(10)  # panel builder enables widgets since state is now ACTIVE

    def enable_wall_division_section(self):
        self._update_sidebar_step(11, ACTIVE)
        # self.show_step(11)  # panel builder enables widgets since state is now ACTIVE

    def enable_pcd_to_lineset_section(self):
        self._update_sidebar_step(12, ACTIVE)
        # self.show_step(12)  # panel builder enables widgets since state is now ACTIVE

    def enable_lineset_to_mesh_section(self):
        self._update_sidebar_step(13, ACTIVE)
        # self.show_step(13)  # panel builder enables widgets since state is now ACTIVE

    def enable_repair_mesh_section(self):
        self._update_sidebar_step(14, ACTIVE)
        # self.show_step(14)  # panel builder enables widgets since state is now ACTIVE

    def enable_cityjson_conversion_section(self):
        self._update_sidebar_step(15, ACTIVE)
        # self.show_step(15)  # panel builder enables widgets since state is now ACTIVE

    def enable_view_pointcloud(self, pointcloud):
        self.view_button.configure(
            state="normal",
            command=lambda: self.view_pointcloud(pointcloud)
        )

    def update_view_pointcloud(self, pointcloud):
        self.view_button.configure(
            state="normal",
            command=lambda: self.view_pointcloud(pointcloud)
        )

    # ── Presets ──────────────────────────────────────────────────────────────

    def _read_presets_config(self):
        config = configparser.ConfigParser()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        presets_file = os.path.join(current_dir, 'presets.ini')
        if not os.path.isfile(presets_file):
            messagebox.showwarning("Waarschuwing", f"Bestand met presets niet gevonden: {presets_file}")
            return None
        config.read(presets_file)
        return config

    def _load_preset_into(self, attr: str, key: str):
        """Load a single preset value into a named entry widget if available."""
        config = self._read_presets_config()
        if config is None:
            return
        try:
            entry: ttk.Entry = getattr(self, attr)
            if config.has_option('Settings', key):
                value = config.get('Settings', key)
                prev_state = str(entry.cget("state"))
                entry.configure(state="normal")
                entry.delete(0, "end")
                entry.insert(0, value)
                entry.configure(state=prev_state)
        except Exception:
            pass

    def load_presets(self):
        """Load all presets at startup — widgets may not all exist yet, so use headless version."""
        self.load_presets_headless()

    def load_presets_headless(self):
        """Load preset values into all currently-existing entry widgets."""
        config = self._read_presets_config()
        if config is None:
            return

        entries_and_keys = [
            ("points_per_cm_entry",              "points_per_cm"),          # noqa: E241
            ("neighbour_amount_entry",            "neighbour_amount"),      # noqa: E241
            ("std_ratio_entry",                   "std_ratio"),             # noqa: E241
            ("floor_alpha_value_entry",           "alpha_value"),           # noqa: E241
            ("floor_triangle_size_entry",         "triangle_size"),         # noqa: E241
            ("corner_distance_threshold_entry",   "distance_threshold"),    # noqa: E241
            ("slice_height_entry",                "slice_height"),          # noqa: E241
            ("roof_layers_entry",                 "roof_layers"),           # noqa: E241
            ("roof_layer_fatness_entry",          "roof_layer_fatness"),    # noqa: E241
            ("roof_voxel_size_entry",             "roof_voxel_size"),       # noqa: E241
            ("roof_angle_threshold_entry",        "angle_threshold"),       # noqa: E241
            ("roof_merge_radius_entry",           "merge_radius"),          # noqa: E241
            ("wall_search_radius_entry",          "wall_search_radius"),    # noqa: E241
            ("wall_layer_amount_entry",           "wall_layer_amount"),     # noqa: E241
            ("xy_tolerance_entry",               "xy_tolerance"),           # noqa: E241
            ("max_line_length_entry",             "max_line_length"),       # noqa: E241
            ("contour_buffer_entry",              "contour_buffer"),        # noqa: E241
        ]

        for attr, key in entries_and_keys:
            try:
                entry: ttk.Entry = getattr(self, attr)
                if not entry.winfo_exists():
                    continue
                if config.has_option('Settings', key):
                    value = config.get('Settings', key)
                    prev_state = str(entry.cget("state"))
                    entry.configure(state="normal")
                    entry.delete(0, "end")
                    entry.insert(0, value)
                    entry.configure(state=prev_state)
            except AttributeError:
                pass  # Widget doesn't exist yet — will be loaded when panel is built

    # ── Misc ─────────────────────────────────────────────────────────────────

    def show_message(self, title: str, message: str, message_type: str = "info"):
        icons = {"info": "ℹ️", "error": "✖", "warning": "⚠️"}
        btn_styles = {"info": "primary", "error": "danger", "warning": "warning"}

        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self.root)
        try:
            dlg.iconbitmap(resource_path(os.path.join("Source", "support_files", "logo.ico")))
        except Exception:
            pass

        body = ttk.Frame(dlg, padding=(24, 20, 24, 12))
        body.pack(fill="x")
        body.columnconfigure(1, weight=1)

        icon_lbl = ttk.Label(body, text=icons.get(message_type, "ℹ️"), font=("Segoe UI", 18))
        icon_lbl.grid(row=0, column=0, sticky="n", padx=(0, 14))
        ttk.Label(body, text=message, font=("Segoe UI", 10),
                  wraplength=320, justify="left").grid(row=0, column=1, sticky="w")

        btn_row = ttk.Frame(dlg, padding=(24, 0, 24, 18))
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="OK", width=10,
                   bootstyle=btn_styles.get(message_type, "primary"),
                   command=dlg.destroy).pack(side="right")

        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")
        self.root.wait_window(dlg)

    def on_close(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Afsluiten")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self.root)
        try:
            dlg.iconbitmap(resource_path(os.path.join("Source", "support_files", "logo.ico")))
        except Exception:
            pass

        ttk.Label(dlg, text="Weet je zeker dat je wilt afsluiten?",
                  font=("Segoe UI", 10), padding=(20, 20, 20, 10)).pack()

        btn_row = ttk.Frame(dlg, padding=(20, 0, 20, 16))
        btn_row.pack(fill="x")

        confirmed = tk.BooleanVar(value=False)

        ttk.Button(btn_row, text="OK", bootstyle="danger",
                   command=lambda: (confirmed.set(True), dlg.destroy())).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ttk.Button(btn_row, text="Annuleren", bootstyle="secondary-outline",
                   command=dlg.destroy).pack(side="left", expand=True, fill="x")

        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

        self.root.wait_window(dlg)
        if confirmed.get():
            self.root.quit()
            exit()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    root = ttk.Window(themename=DARK_THEME)
    app = App(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
