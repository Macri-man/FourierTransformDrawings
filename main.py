#!/usr/bin/env python3
"""
Fourier Epicycle Simulator (single-file)

Features:
- Freehand drawing
- Full SVG import (M, L, C, Q, Z)
- Parameterized math presets with sliders (example preset included)
- Animated parameters (sinusoidal)
- Fourier epicycles visualization
- Gradient / rainbow trail with motion blur (fading)
- Random gradient palette generator
- Epicycle & trail color customization
- Real-time preview in GUI
- MP4 and GIF export with optional seamless looping
"""

import sys, math, re, json, random, time
import numpy as np
import imageio
import cv2
from xml.dom import minidom

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QSlider, QSpinBox, QCheckBox, QComboBox, QFileDialog, QDialog,
    QColorDialog, QMessageBox, QInputDialog, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen

# ----------------------------
# Utility: Bézier samplers
# ----------------------------
def lerp(a, b, t):
    return a + (b - a) * t

def cubic_bezier_points(p0, p1, p2, p3, n):
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        x = ((1-t)**3)*p0[0] + 3*((1-t)**2)*t*p1[0] + 3*(1-t)*(t**2)*p2[0] + (t**3)*p3[0]
        y = ((1-t)**3)*p0[1] + 3*((1-t)**2)*t*p1[1] + 3*(1-t)*(t**2)*p2[1] + (t**3)*p3[1]
        pts.append((x,y))
    return pts

def quadratic_bezier_points(p0, p1, p2, n):
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
        pts.append((x,y))
    return pts

def line_points(p0, p1, n):
    return [(lerp(p0[0], p1[0], t), lerp(p0[1], p1[1], t)) for t in np.linspace(0.0, 1.0, n)]

# ----------------------------
# DFT
# ----------------------------
def compute_fourier_coeffs(points):
    # points: list of (x,y) or complex
    if len(points) == 0:
        return []
    pts = np.array([complex(p[0], p[1]) if not isinstance(p, complex) else p for p in points])
    N = len(pts)
    coeffs = []
    # Compute centered DFT indices to capture negative frequencies
    for k in range(-N//2, N//2):
        exp = np.exp(-2j * np.pi * k * np.arange(N) / N)
        c = np.sum(pts * exp) / N
        coeffs.append((k, c))
    # sort by magnitude descending (largest epicycles first)
    coeffs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return coeffs

# ----------------------------
# Main Widget
# ----------------------------
class FourierWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Epicycle Simulator")
        self.resize(1200, 820)

        # Canvas
        self.canvas_w = 1000
        self.canvas_h = 700
        self.canvas = QLabel()
        self.pixmap = QPixmap(self.canvas_w, self.canvas_h)
        self.pixmap.fill(Qt.GlobalColor.black)
        self.canvas.setPixmap(self.pixmap)

        # State
        self.time = 0.0
        self.playing = True
        self.animate_params = True
        self.trail = []            # list of (x,y) floats
        self.trail_max = 1000
        self.coeffs = []           # (freq, complex coeff)
        self.current_preset = None
        self.param_sliders = {}    # name -> QSlider
        self.epicycle_width = 1
        self.trail_width = 2
        self.epicycle_color = QColor(160,160,160)
        self.single_trail_color = None  # QColor or None -> use gradient/rainbow
        self.gradient_colors = [QColor("red"), QColor("yellow"), QColor("green"), QColor("cyan"), QColor("blue"), QColor("magenta")]
        self.trail_palette = None  # precomputed list of QColor if user set gradient
        self.trail_palette_len = 256

        # Drawing / SVG
        self.drawing_points = []   # raw canvas coords (x,y)
        self.drawing_active = False

        # UI
        self.setup_ui()

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33ms => ~30 FPS

        # Example math preset
        self.load_default_presets()
        self.select_preset_index(0)

    # ----------------------------
    # UI creation
    # ----------------------------
    def setup_ui(self):
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.canvas)

        # Controls row 1
        row1 = QHBoxLayout()
        self.play_btn = QPushButton("Play/Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        row1.addWidget(self.play_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_canvas)
        row1.addWidget(self.clear_btn)

        self.draw_btn = QPushButton("Draw Shape")
        self.draw_btn.clicked.connect(self.activate_drawing)
        row1.addWidget(self.draw_btn)

        self.svg_btn = QPushButton("Import SVG")
        self.svg_btn.clicked.connect(self.import_svg)
        row1.addWidget(self.svg_btn)

        self.preset_dropdown = QComboBox()
        row1.addWidget(self.preset_dropdown)

        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.clicked.connect(self.save_current_preset_to_db)
        row1.addWidget(self.save_preset_btn)

        main_layout.addLayout(row1)

        # Controls row 2 (parameters + visuals)
        row2 = QHBoxLayout()
        self.params_area = QVBoxLayout()
        row2.addLayout(self.params_area)

        visuals = QVBoxLayout()
        # Epicycle width slider
        ew_layout = QHBoxLayout()
        ew_layout.addWidget(QLabel("Epicycle Width"))
        self.epi_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.epi_width_slider.setRange(1,8)
        self.epi_width_slider.setValue(self.epicycle_width)
        self.epi_width_slider.valueChanged.connect(lambda v: setattr(self, 'epicycle_width', v))
        ew_layout.addWidget(self.epi_width_slider)
        visuals.addLayout(ew_layout)

        # Trail width
        tw_layout = QHBoxLayout()
        tw_layout.addWidget(QLabel("Trail Width"))
        self.trail_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.trail_width_slider.setRange(1,10)
        self.trail_width_slider.setValue(self.trail_width)
        self.trail_width_slider.valueChanged.connect(lambda v: setattr(self, 'trail_width', v))
        tw_layout.addWidget(self.trail_width_slider)
        visuals.addLayout(tw_layout)

        # Epicycle color button
        self.epi_color_btn = QPushButton("Epicycle Color")
        self.epi_color_btn.clicked.connect(self.choose_epicycle_color)
        visuals.addWidget(self.epi_color_btn)

        # Single trail color (optional)
        self.trail_color_btn = QPushButton("Single Trail Color (toggle)")
        self.trail_color_btn.clicked.connect(self.choose_single_trail_color)
        visuals.addWidget(self.trail_color_btn)

        # Gradient buttons
        self.gradient_btn = QPushButton("Edit Gradient Colors")
        self.gradient_btn.clicked.connect(self.edit_gradient_colors)
        visuals.addWidget(self.gradient_btn)

        self.random_grad_btn = QPushButton("Random Gradient")
        self.random_grad_btn.clicked.connect(self.randomize_gradient)
        visuals.addWidget(self.random_grad_btn)

        row2.addLayout(visuals)
        main_layout.addLayout(row2)

        # Controls row 3 (export + animate)
        row3 = QHBoxLayout()
        self.animate_chk = QCheckBox("Animate Params")
        self.animate_chk.setChecked(True)
        self.animate_chk.stateChanged.connect(lambda s: setattr(self, 'animate_params', s==Qt.CheckState.Checked))
        row3.addWidget(self.animate_chk)

        self.export_btn = QPushButton("Export (MP4/GIF)")
        self.export_btn.clicked.connect(self.show_export_dialog)
        row3.addWidget(self.export_btn)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        row3.addWidget(self.progress)

        main_layout.addLayout(row3)

        self.setLayout(main_layout)

    # ----------------------------
    # Preset DB (memory + file)
    # ----------------------------
    def load_default_presets(self):
        # Prepopulate dropdown and internal presets (support params)
        self.presets = [
            {
                "name": "Rose Curve (k adjustable)",
                "mode": "polar",
                "expr_r": "math.cos(k*t)",
                "params": {"k":[3,1,10]},
                "range":[0, 2*math.pi]
            },
            {
                "name": "Lissajous (a,b adjustable)",
                "mode": "parametric",
                "expr_x": "math.cos(a*t)",
                "expr_y": "math.sin(b*t)",
                "params": {"a":[3,1,10], "b":[2,1,10]},
                "range":[0, 2*math.pi]
            },
            {
                "name": "Cardioid",
                "mode": "polar",
                "expr_r": "1 - math.cos(t)",
                "params": {},
                "range":[0, 2*math.pi]
            },
        ]
        self.preset_dropdown.clear()
        for p in self.presets:
            self.preset_dropdown.addItem(p["name"])
        self.preset_dropdown.currentIndexChanged.connect(self.preset_selected)

    def preset_selected(self, idx):
        if idx < 0 or idx >= len(self.presets):
            return
        self.select_preset_index(idx)

    def select_preset_index(self, idx):
        self.current_preset = self.presets[idx]
        # rebuild sliders UI
        # clear current
        for i in reversed(range(self.params_area.count())):
            item = self.params_area.itemAt(i)
            if item is None:
                continue
            w = item.widget()
            if w:
                w.setParent(None)
        self.param_sliders = {}
        if "params" in self.current_preset:
            for pname, (default, minv, maxv) in self.current_preset["params"].items():
                lbl = QLabel(f"{pname} ({minv}–{maxv})")
                sld = QSlider(Qt.Orientation.Horizontal)
                sld.setMinimum(int(minv*10))
                sld.setMaximum(int(maxv*10))
                sld.setValue(int(default*10))
                sld.valueChanged.connect(self.update_points)
                self.params_area.addWidget(lbl)
                self.params_area.addWidget(sld)
                self.param_sliders[pname] = sld
        self.update_points()

    def save_current_preset_to_db(self):
        if not self.current_preset:
            return
        name, ok = QInputDialog.getText(self, "Save Preset", "Name for preset:")
        if not ok or not name:
            return
        p = self.current_preset.copy()
        p["name"] = name
        # save sliders if any
        if "params" in p:
            for k,s in self.param_sliders.items():
                p["params"][k][0] = s.value() / 10.0
        # append to file math_shapes.json
        try:
            db = []
            try:
                with open("math_shapes.json","r") as f:
                    db = json.load(f)
            except FileNotFoundError:
                db = []
            db.append(p)
            with open("math_shapes.json","w") as f:
                json.dump(db, f, indent=2)
            QMessageBox.information(self, "Saved", f"Preset '{name}' saved to math_shapes.json")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Couldn't save preset: {e}")

    # ----------------------------
    # Evaluate (safe-ish)
    # ----------------------------
    def evaluate_expr(self, expr, local_vars):
        # Provide math module only
        try:
            return eval(expr, {"__builtins__":{},"math":math,"np":np}, local_vars)
        except Exception as e:
            # print on console for debugging, continue with 0
            print("Eval error:", e, "expr:", expr, "locals:", local_vars)
            return 0.0

    # ----------------------------
    # Update points from either preset or drawing
    # ----------------------------
    def update_points(self):
        if self.drawing_points:
            pts = list(self.drawing_points)
            self.compute_fourier_from_points(pts)
            return

        p = self.current_preset
        if not p:
            return
        tmin, tmax = p.get("range", [0, 2*math.pi])
        t_vals = np.linspace(tmin, tmax, 500)
        points = []
        local = {k: (s.value()/10.0) for k,s in self.param_sliders.items()}
        if p["mode"] == "polar":
            for t in t_vals:
                r = self.evaluate_expr(p["expr_r"], {**local, "t": t})
                x = r * math.cos(t)
                y = r * math.sin(t)
                points.append((x, y))
        else:  # parametric
            for t in t_vals:
                x = self.evaluate_expr(p["expr_x"], {**local, "t": t})
                y = self.evaluate_expr(p["expr_y"], {**local, "t": t})
                points.append((x, y))
        # Normalize points to canvas (centered, scaled)
        points = self.normalize_points(points)
        self.compute_fourier_from_points(points)

    # ----------------------------
    # Normalization helper
    # ----------------------------
    def normalize_points(self, pts):
        if not pts:
            return []
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        w = maxx - minx if maxx - minx != 0 else 1.0
        h = maxy - miny if maxy - miny != 0 else 1.0
        scale = min(self.canvas_w / w, self.canvas_h / h) * 0.8
        cx = self.canvas_w / 2
        cy = self.canvas_h / 2
        cx_shape = (minx + maxx) / 2
        cy_shape = (miny + maxy) / 2
        out = [((x - cx_shape) * scale + cx, (y - cy_shape) * scale + cy) for (x,y) in pts]
        return out

    # ----------------------------
    # Compute DFT from points list
    # ----------------------------
    def compute_fourier_from_points(self, pts):
        if not pts:
            self.coeffs = []
            return
        # Convert to evenly sampled sequence if needed
        # Here we assume pts is already a list of canvas points
        self.coeffs = compute_fourier_coeffs(pts)
        self.trail = []

    # ----------------------------
    # Drawing / Rendering
    # ----------------------------
    def draw(self, epicycle_color=None, gradient=None):
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(0, 0, self.canvas_w, self.canvas_h, Qt.GlobalColor.black)

        cx = self.canvas_w / 2
        cy = self.canvas_h / 2

        if not self.coeffs:
            painter.end()
            self.canvas.setPixmap(self.pixmap)
            return

        x = 0.0
        y = 0.0
        # Draw epicycles (largest to smallest)
        for freq, c in self.coeffs:
            prev_x, prev_y = x, y
            ang = 2 * math.pi * freq * self.time + np.angle(c)
            radius = abs(c)
            x += radius * math.cos(ang)
            y += radius * math.sin(ang)
            painter.setPen(QPen(epicycle_color or self.epicycle_color, self.epicycle_width))
            painter.drawEllipse(QPointF(cx + prev_x, cy + prev_y), float(radius), float(radius))
            painter.drawLine(QPointF(cx + prev_x, cy + prev_y), QPointF(cx + x, cy + y))

        # Trail handling (motion blur / fade)
        self.trail.append((cx + x, cy + y))
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)

        palette = self.trail_palette
        if palette is None:
            # create palette from gradient_colors
            palette = []
            ncolors = len(self.gradient_colors)
            for i in range(self.trail_palette_len):
                t = i / (self.trail_palette_len - 1) * (ncolors - 1)
                idx = int(t)
                frac = t - idx
                c1 = self.gradient_colors[idx]
                c2 = self.gradient_colors[min(idx + 1, ncolors - 1)]
                r = int(c1.red() * (1 - frac) + c2.red() * frac)
                g = int(c1.green() * (1 - frac) + c2.green() * frac)
                b = int(c1.blue() * (1 - frac) + c2.blue() * frac)
                palette.append(QColor(r, g, b))

        L = len(self.trail)
        for i in range(1, L):
            # choose color
            if self.single_trail_color:
                color = self.single_trail_color
            else:
                idx = int((i / max(1, L-1)) * (len(palette)-1))
                # animate palette over time (hue shift) if desired:
                # idx = int((idx + self.time*20) % len(palette))
                color = palette[idx]
            alpha = int(255 * (i / L))  # older points more transparent (fade out)
            pen = QPen(QColor(color.red(), color.green(), color.blue(), alpha), self.trail_width)
            painter.setPen(pen)
            p0 = QPointF(*self.trail[i-1])
            p1 = QPointF(*self.trail[i])
            painter.drawLine(p0, p1)

        painter.end()
        self.canvas.setPixmap(self.pixmap)

    # ----------------------------
    # Frame update (animation)
    # ----------------------------
    def update_frame(self):
        if self.playing:
            self.time += 0.03
        if self.animate_params and self.current_preset and "params" in self.current_preset:
            # animate sliders sinusoidally between min and max
            for i, (pname, slider) in enumerate(self.param_sliders.items()):
                default, minv, maxv = self.current_preset["params"][pname]
                amp = (maxv - minv) / 2.0
                offset = (maxv + minv) / 2.0
                val = offset + amp * math.sin(0.5 * self.time + i)
                slider.setValue(int(val * 10))
            self.update_points()
        else:
            # If not animating params but sliders exist and were moved, keep points updated
            # (slider.valueChanged already calls update_points)
            pass
        self.draw()

    # ----------------------------
    # Freehand drawing mouse handlers
    # ----------------------------
    def activate_drawing(self):
        self.drawing_points = []
        self.drawing_active = True
        self.canvas.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if self.drawing_active and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.drawing_points = [(pos.x(), pos.y())]
            self.update_canvas_drawing()

    def mouseMoveEvent(self, event):
        if self.drawing_active and (event.buttons() & Qt.MouseButton.LeftButton):
            pos = event.position()
            self.drawing_points.append((pos.x(), pos.y()))
            self.update_canvas_drawing()

    def mouseReleaseEvent(self, event):
        if self.drawing_active and event.button() == Qt.MouseButton.LeftButton:
            self.drawing_active = False
            # compute fourier from the drawn points (resample)
            pts = self.resample_points(self.drawing_points, 500)
            self.drawing_points = pts
            self.compute_fourier_from_points(pts)
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def update_canvas_drawing(self):
        painter = QPainter(self.pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(0,0,self.canvas_w,self.canvas_h, Qt.GlobalColor.black)
        pen = QPen(Qt.GlobalColor.white, 2)
        painter.setPen(pen)
        for i in range(1, len(self.drawing_points)):
            p0 = QPointF(*self.drawing_points[i-1])
            p1 = QPointF(*self.drawing_points[i])
            painter.drawLine(p0, p1)
        painter.end()
        self.canvas.setPixmap(self.pixmap)

    def resample_points(self, pts, n):
        # simple uniform resampling along cumulative distance
        if len(pts) == 0:
            return []
        arr = np.array(pts)
        diffs = arr[1:] - arr[:-1]
        seg_lens = np.hypot(diffs[:,0], diffs[:,1])
        cum = np.concatenate(([0.0], np.cumsum(seg_lens)))
        total = cum[-1] if cum[-1] > 0 else 1.0
        new_t = np.linspace(0, total, n)
        xs = np.interp(new_t, cum, arr[:,0])
        ys = np.interp(new_t, cum, arr[:,1])
        return list(zip(xs.tolist(), ys.tolist()))

    # ----------------------------
    # SVG import (full support for M,L,C,Q,Z)
    # ----------------------------
    def import_svg(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open SVG", "", "SVG Files (*.svg)")
        if not fname:
            return
        try:
            doc = minidom.parse(fname)
            path_elems = doc.getElementsByTagName("path")
            all_pts = []
            for path in path_elems:
                d = path.getAttribute("d")
                pts = self.svg_path_to_points(d, num_points=1000)
                all_pts.extend(pts)
            doc.unlink()
            if not all_pts:
                QMessageBox.warning(self, "SVG Import", "No usable path points found in SVG.")
                return
            # normalize to canvas
            pts_norm = self.normalize_points(all_pts)
            pts_sampled = self.resample_points(pts_norm, 800)
            self.drawing_points = pts_sampled
            self.compute_fourier_from_points(pts_sampled)
        except Exception as e:
            QMessageBox.warning(self, "SVG Import Error", str(e))

    def svg_path_to_points(self, d_attr, num_points=500):
        # Tokenize by commands
        tokens = re.findall(r'([MLCQZmlcqz])([^MLCQZmlcqz]*)', d_attr)
        cur = (0.0, 0.0)
        start = (0.0, 0.0)
        pts = []
        for cmd, args in tokens:
            cmdu = cmd.upper()
            vals = list(map(float, re.findall(r'[-+]?[0-9]*\.?[0-9]+', args)))
            # Process absolute commands - relative could be added if necessary.
            if cmdu == 'M':
                # move
                for i in range(0, len(vals), 2):
                    cur = (vals[i], vals[i+1])
                    start = cur
                    pts.append(cur)
            elif cmdu == 'L':
                for i in range(0, len(vals), 2):
                    nxt = (vals[i], vals[i+1])
                    pts.extend(line_points(cur, nxt, max(2, int(num_points/20))))
                    cur = nxt
            elif cmdu == 'C':
                # Cubic bezier: x1,y1, x2,y2, x3,y3
                for i in range(0, len(vals), 6):
                    p1 = (vals[i], vals[i+1])
                    p2 = (vals[i+2], vals[i+3])
                    p3 = (vals[i+4], vals[i+5])
                    bez = cubic_bezier_points(cur, p1, p2, p3, max(4, int(num_points/30)))
                    pts.extend(bez)
                    cur = p3
            elif cmdu == 'Q':
                # Quadratic bezier: x1,y1, x2,y2
                for i in range(0, len(vals), 4):
                    p1 = (vals[i], vals[i+1])
                    p2 = (vals[i+2], vals[i+3])
                    bez = quadratic_bezier_points(cur, p1, p2, max(4, int(num_points/30)))
                    pts.extend(bez)
                    cur = p2
            elif cmdu == 'Z':
                # close path: line to start
                pts.extend(line_points(cur, start, max(2, int(num_points/20))))
                cur = start
            else:
                # unsupported command; ignore
                pass
        return pts

    # ----------------------------
    # Visual controls
    # ----------------------------
    def choose_epicycle_color(self):
        c = QColorDialog.getColor(self.epicycle_color, self, "Select Epicycle Color")
        if c.isValid():
            self.epicycle_color = c

    def choose_single_trail_color(self):
        c = QColorDialog.getColor(self.single_trail_color or QColor(255,255,255), self, "Select Trail Color (Cancel to use gradient)")
        if c.isValid():
            self.single_trail_color = c
        else:
            # cancel -> toggle off
            self.single_trail_color = None

    def edit_gradient_colors(self):
        # present sequential dialogs to edit each color
        newcols = []
        for i, c in enumerate(self.gradient_colors):
            chosen = QColorDialog.getColor(c, self, f"Edit gradient color {i+1}")
            if chosen.isValid():
                newcols.append(chosen)
            else:
                newcols.append(c)
        self.gradient_colors = newcols
        self.trail_palette = None  # recompute palette on next draw

    def randomize_gradient(self):
        self.gradient_colors = [QColor(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(6)]
        self.trail_palette = None

    # ----------------------------
    # Clear canvas
    # ----------------------------
    def clear_canvas(self):
        self.drawing_points = []
        self.coeffs = []
        self.trail = []
        self.pixmap.fill(Qt.GlobalColor.black)
        self.canvas.setPixmap(self.pixmap)

    # ----------------------------
    # Export: unified dialog and recording functions
    # ----------------------------
    def show_export_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Export Animation")
        layout = QVBoxLayout()

        layout.addWidget(QLabel("File Type:"))
        ft = QComboBox()
        ft.addItems(["MP4", "GIF"])
        layout.addWidget(ft)

        layout.addWidget(QLabel("Duration (seconds):"))
        dur = QSpinBox()
        dur.setRange(1, 120)
        dur.setValue(8)
        layout.addWidget(dur)

        layout.addWidget(QLabel("FPS:"))
        fps_spin = QSpinBox()
        fps_spin.setRange(1, 60)
        fps_spin.setValue(30)
        layout.addWidget(fps_spin)

        layout.addWidget(QLabel("Width (px) for MP4:"))
        width_spin = QSpinBox()
        width_spin.setRange(100, 3840)
        width_spin.setValue(self.canvas_w)
        layout.addWidget(width_spin)

        seamless_cb = QCheckBox("Seamless loop (uses sine animation)")
        seamless_cb.setChecked(True)
        layout.addWidget(seamless_cb)

        # preview thumbnail
        preview_label = QLabel()
        preview_label.setFixedSize(240, 160)
        layout.addWidget(QLabel("Preview (first frames):"))
        layout.addWidget(preview_label)

        btn_preview = QPushButton("Generate Preview")
        layout.addWidget(btn_preview)

        buttons = QHBoxLayout()
        ok = QPushButton("Export")
        cancel = QPushButton("Cancel")
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        dlg.setLayout(layout)

        def make_preview():
            preview_frames = self.generate_preview_frames(frames=12, w=240, h=160, seamless=seamless_cb.isChecked())
            if preview_frames:
                # cycle first frame in label (statically)
                preview_label.setPixmap(preview_frames[0])

        btn_preview.clicked.connect(make_preview)
        cancel.clicked.connect(dlg.close)

        def do_export():
            fname, _ = QFileDialog.getSaveFileName(self, "Save Animation", "", "MP4 Files (*.mp4);;GIF Files (*.gif)")
            if not fname:
                return
            dtype = ft.currentText()
            duration = dur.value()
            fps = fps_spin.value()
            width = width_spin.value()
            seamless = seamless_cb.isChecked()
            # show progress bar
            self.progress.setVisible(True)
            self.progress.setValue(0)
            QApplication.processEvents()
            try:
                if dtype == "MP4":
                    if not fname.lower().endswith(".mp4"):
                        fname += ".mp4"
                    if seamless:
                        self.record_seamless_video(fname, duration=duration, fps=fps, width=width)
                    else:
                        self.record_video(fname, duration=duration, fps=fps, width=width)
                else:
                    if not fname.lower().endswith(".gif"):
                        fname += ".gif"
                    if seamless:
                        self.record_seamless_gif(fname, duration=duration, fps=fps)
                    else:
                        self.record_gif(fname, duration=duration, fps=fps)
                QMessageBox.information(self, "Export", f"Saved to {fname}")
            finally:
                self.progress.setVisible(False)
            dlg.close()

        ok.clicked.connect(do_export)
        dlg.exec()

    def generate_preview_frames(self, frames=12, w=240, h=160, seamless=True):
        orig_time = self.time
        orig_trail = list(self.trail)
        frames_out = []
        for i in range(frames):
            if seamless:
                t_norm = 2 * math.pi * (i / frames)
                # animate params by setting sliders to sine value
                if self.current_preset and "params" in self.current_preset:
                    for j, (pname, sld) in enumerate(self.param_sliders.items()):
                        default, minv, maxv = self.current_preset["params"][pname]
                        amp = (maxv - minv) / 2.0
                        offset = (maxv + minv) / 2.0
                        val = offset + amp * math.sin(t_norm + j)
                        sld.setValue(int(val * 10))
                    self.update_points()
                self.time = i / 30.0
            else:
                self.time = i / 30.0
            self.draw()
            scaled = self.pixmap.scaled(w, h)
            frames_out.append(scaled)
        self.time = orig_time
        self.trail = orig_trail
        return frames_out

    # ----------------------------
    # Recording functions
    # ----------------------------
    def record_gif(self, filename, duration=5, fps=30):
        frames = []
        orig_time = self.time
        orig_trail = list(self.trail)
        total = duration * fps
        for i in range(total):
            self.time = i / fps
            # animate params (non-seamless): use same variant as current slider positions
            if self.current_preset and "params" in self.current_preset and self.animate_params:
                for j, (pname, sld) in enumerate(self.param_sliders.items()):
                    default, minv, maxv = self.current_preset["params"][pname]
                    amp = (maxv - minv) / 2.0
                    offset = (maxv + minv) / 2.0
                    val = offset + amp * math.sin(0.5 * self.time + j)
                    sld.setValue(int(val * 10))
                self.update_points()
            self.draw()
            img = self.pixmap.toImage()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4)
            # convert RGBA to RGB
            frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            frames.append(frame)
            # update progress
            if self.progress.isVisible():
                self.progress.setValue(int(100 * (i+1) / total))
                QApplication.processEvents()
        imageio.mimsave(filename, frames, fps=fps)
        self.time = orig_time
        self.trail = orig_trail

    def record_seamless_gif(self, filename, duration=5, fps=30):
        frames = []
        orig_time = self.time
        orig_trail = list(self.trail)
        total = duration * fps
        for i in range(total):
            t_norm = 2 * math.pi * (i / total)
            self.time = i / fps
            # animate parameters seamlessly (sine full period)
            if self.current_preset and "params" in self.current_preset:
                for j, (pname, sld) in enumerate(self.param_sliders.items()):
                    default, minv, maxv = self.current_preset["params"][pname]
                    amp = (maxv - minv) / 2.0
                    offset = (maxv + minv) / 2.0
                    val = offset + amp * math.sin(t_norm + j)
                    sld.setValue(int(val * 10))
                self.update_points()
            self.draw()
            img = self.pixmap.toImage()
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4)
            frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            frames.append(frame)
            if self.progress.isVisible():
                self.progress.setValue(int(100 * (i+1) / total))
                QApplication.processEvents()
        imageio.mimsave(filename, frames, fps=fps)
        self.time = orig_time
        self.trail = orig_trail

    def record_video(self, filename, duration=5, fps=30, width=None):
        orig_time = self.time
        orig_trail = list(self.trail)
        if width is None:
            width = self.canvas_w
        height = int(width * self.canvas_h / self.canvas_w)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        total = duration * fps
        for i in range(total):
            self.time = i / fps
            # animate params if requested (non-seamless)
            if self.current_preset and "params" in self.current_preset and self.animate_params:
                for j, (pname, sld) in enumerate(self.param_sliders.items()):
                    default, minv, maxv = self.current_preset["params"][pname]
                    amp = (maxv - minv) / 2.0
                    offset = (maxv + minv) / 2.0
                    val = offset + amp * math.sin(0.5 * self.time + j)
                    sld.setValue(int(val * 10))
                self.update_points()
            self.draw()
            img = self.pixmap.toImage().scaled(width, height)
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4)
            frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            out.write(frame)
            if self.progress.isVisible():
                self.progress.setValue(int(100 * (i+1) / total))
                QApplication.processEvents()
        out.release()
        self.time = orig_time
        self.trail = orig_trail

    def record_seamless_video(self, filename, duration=5, fps=30, width=None):
        orig_time = self.time
        orig_trail = list(self.trail)
        if width is None:
            width = self.canvas_w
        height = int(width * self.canvas_h / self.canvas_w)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        total = duration * fps
        for i in range(total):
            t_norm = 2 * math.pi * (i / total)
            self.time = i / fps
            if self.current_preset and "params" in self.current_preset:
                for j, (pname, sld) in enumerate(self.param_sliders.items()):
                    default, minv, maxv = self.current_preset["params"][pname]
                    amp = (maxv - minv) / 2.0
                    offset = (maxv + minv) / 2.0
                    val = offset + amp * math.sin(t_norm + j)
                    sld.setValue(int(val * 10))
                self.update_points()
            self.draw()
            img = self.pixmap.toImage().scaled(width, height)
            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.array(ptr, dtype=np.uint8).reshape(img.height(), img.width(), 4)
            frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            out.write(frame)
            if self.progress.isVisible():
                self.progress.setValue(int(100 * (i+1) / total))
                QApplication.processEvents()
        out.release()
        self.time = orig_time
        self.trail = orig_trail

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # ensure DPI rounding policy before creating app
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass
    app = QApplication(sys.argv)
    w = FourierWidget()
    w.show()
    sys.exit(app.exec())
