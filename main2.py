# main.py
import sys
import math
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QFileDialog, QLabel, QSlider, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QMouseEvent

# -------------------------
# Utility: Fourier helper
# -------------------------
def compute_fourier(signal, num_terms=100):
    """Compute Fourier coefficients (centered frequency indices).
       Returns list of (k, c) sorted by decreasing magnitude (keeps negative & positive freqs)."""
    N = len(signal)
    if N == 0:
        return []
    # Use direct DFT (simple). We will produce coefficients for k in [-M..M]
    # But to keep implementation simple and stable, use fft and pair with freqs
    fft = np.fft.fft(signal) / N
    freqs = np.fft.fftfreq(N, d=1.0/N)  # integer indices effectively
    coeffs = list(zip(freqs.astype(int).tolist(), fft.tolist()))
    # sort by magnitude descending to visualize most important terms first
    coeffs.sort(key=lambda kc: -abs(kc[1]))
    return coeffs

# -------------------------
# Simple SVG path parser (M,L,C,Q,A,Z)
# Note: This parser is basic but handles multiple path elements.
# For arcs ('A') we fallback to linear sampling if arc code isn't available.
# -------------------------
def _tokenize_path(d):
    # Tokens: single letters or numbers (allow decimals & negatives)
    toks = re.findall(r"[MLCQAZmlcqaz]|-?\d+\.?\d*(?:e[+-]?\d+)?", d)
    return toks

def parse_svg_path(d, num_samples=200):
    toks = _tokenize_path(d)
    i = 0
    cur = complex(0,0)
    start = complex(0,0)
    pts = []

    def to_c(x, y):
        return complex(float(x), float(y))

    while i < len(toks):
        cmd = toks[i]; i += 1
        if cmd == 'M':
            x, y = toks[i], toks[i+1]; i += 2
            cur = to_c(x,y); start = cur
            pts.append(cur)
        elif cmd == 'm':
            x, y = toks[i], toks[i+1]; i += 2
            cur += to_c(x,y); start = cur
            pts.append(cur)
        elif cmd == 'L':
            x, y = toks[i], toks[i+1]; i += 2
            new = to_c(x,y)
            for t in np.linspace(0,1, max(2, num_samples//20)):
                pts.append(cur*(1-t) + new*t)
            cur = new
        elif cmd == 'l':
            x, y = toks[i], toks[i+1]; i += 2
            new = cur + to_c(x,y)
            for t in np.linspace(0,1, max(2, num_samples//20)):
                pts.append(cur*(1-t) + new*t)
            cur = new
        elif cmd == 'C':
            x1,y1,x2,y2,x3,y3 = toks[i:i+6]; i+=6
            p0 = cur
            p1 = to_c(x1,y1)
            p2 = to_c(x2,y2)
            p3 = to_c(x3,y3)
            for t in np.linspace(0,1, max(5, num_samples//20)):
                tt = t
                b = (1-tt)**3 * p0 + 3*(1-tt)**2*tt*p1 + 3*(1-tt)*tt**2*p2 + tt**3*p3
                pts.append(b)
            cur = p3
        elif cmd == 'c':
            x1,y1,x2,y2,x3,y3 = toks[i:i+6]; i+=6
            p0 = cur
            p1 = cur + to_c(x1,y1)
            p2 = cur + to_c(x2,y2)
            p3 = cur + to_c(x3,y3)
            for t in np.linspace(0,1, max(5, num_samples//20)):
                tt = t
                b = (1-tt)**3 * p0 + 3*(1-tt)**2*tt*p1 + 3*(1-tt)*tt**2*p2 + tt**3*p3
                pts.append(b)
            cur = p3
        elif cmd == 'Q':
            x1,y1,x2,y2 = toks[i:i+4]; i+=4
            p0 = cur
            p1 = to_c(x1,y1)
            p2 = to_c(x2,y2)
            for t in np.linspace(0,1, max(5, num_samples//20)):
                tt = t
                b = (1-tt)**2*p0 + 2*(1-tt)*tt*p1 + tt**2*p2
                pts.append(b)
            cur = p2
        elif cmd == 'q':
            x1,y1,x2,y2 = toks[i:i+4]; i+=4
            p0 = cur
            p1 = cur + to_c(x1,y1)
            p2 = cur + to_c(x2,y2)
            for t in np.linspace(0,1, max(5, num_samples//20)):
                tt = t
                b = (1-tt)**2*p0 + 2*(1-tt)*tt*p1 + tt**2*p2
                pts.append(b)
            cur = p2
        elif cmd == 'A' or cmd == 'a':
            # Simplified: just linearly interpolate to endpoint (safe fallback)
            # A full arc implementation is more verbose (we covered it earlier).
            # Read 7 params
            rx = float(toks[i]); ry = float(toks[i+1]); angle = float(toks[i+2])
            laf = float(toks[i+3]); sf = float(toks[i+4])
            x = toks[i+5]; y = toks[i+6]; i += 7
            new = to_c(x,y) if cmd == 'A' else cur + to_c(x,y)
            for t in np.linspace(0,1, max(5, num_samples//20)):
                pts.append(cur*(1-t) + new*t)
            cur = new
        elif cmd in ('Z', 'z'):
            # close path
            for t in np.linspace(0,1, max(3, num_samples//20)):
                pts.append(cur*(1-t) + start*t)
            cur = start
        else:
            # unknown token -> skip
            pass

    return np.array(pts, dtype=complex)

# -------------------------
# Helper: save multi-stroke SVG
# -------------------------
def save_multistroke_svg(strokes, filename, canvas_size=(1000,1000)):
    """
    strokes: list of sequences of complex points
    """
    if not strokes:
        return
    width, height = canvas_size
    # Build SVG path elements
    paths = []
    for stroke in strokes:
        if len(stroke) == 0:
            continue
        d = f"M {stroke[0].real:.3f},{stroke[0].imag:.3f} "
        for p in stroke[1:]:
            d += f"L {p.real:.3f},{p.imag:.3f} "
        # do not force close; keep it as drawn
        paths.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="2"/>')
    svg = '<?xml version="1.0" encoding="UTF-8"?>\n'
    svg += f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" version="1.1">\n'
    svg += "\n".join(paths)
    svg += "\n</svg>"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg)

# -------------------------
# Main QWidget
# -------------------------
class FourierWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fourier Multi-Stroke Epicycles")
        self.resize(1100, 800)

        # Layout
        main = QVBoxLayout(self)
        ctrl = QHBoxLayout()
        main.addLayout(ctrl)

        # Buttons / controls
        self.load_svg_btn = QPushButton("Load SVG")
        self.load_svg_btn.clicked.connect(self.on_load_svg)
        ctrl.addWidget(self.load_svg_btn)

        self.save_svg_btn = QPushButton("Save Drawing")
        self.save_svg_btn.clicked.connect(self.on_save_svg)
        ctrl.addWidget(self.save_svg_btn)

        self.clear_btn = QPushButton("Clear Strokes")
        self.clear_btn.clicked.connect(self.clear_strokes)
        ctrl.addWidget(self.clear_btn)

        self.draw_mode_btn = QPushButton("Drawing Mode")
        self.draw_mode_btn.setCheckable(True)
        self.draw_mode_btn.clicked.connect(self.toggle_drawing_mode)
        ctrl.addWidget(self.draw_mode_btn)

        self.new_stroke_btn = QPushButton("Start New Stroke")
        self.new_stroke_btn.clicked.connect(self.start_new_stroke)
        ctrl.addWidget(self.new_stroke_btn)

        self.combine_checkbox = QCheckBox("Combine strokes into single Fourier")
        self.combine_checkbox.setChecked(False)
        ctrl.addWidget(self.combine_checkbox)

        ctrl.addWidget(QLabel("Terms:"))
        self.term_slider = QSlider(Qt.Orientation.Horizontal)
        self.term_slider.setMinimum(10)
        self.term_slider.setMaximum(800)
        self.term_slider.setValue(200)
        self.term_slider.setTickInterval(10)
        ctrl.addWidget(self.term_slider)

        self.playing = True
        self.play_btn = QPushButton("Pause")
        self.play_btn.clicked.connect(self.toggle_play)
        ctrl.addWidget(self.play_btn)

        # state
        self.drawing_mode = False
        self.current_stroke = []        # points of stroke being drawn (complex)
        self.strokes = []               # list of strokes (each a complex numpy array)
        self.stroke_coeffs = []         # list of lists of Fourier coeffs per stroke
        self.combined_coeffs = []       # combined if combine mode
        self.trails = []                # trails for each stroke (list of complex)
        self.max_trail = 800
        self.hue_offset = 0.0
        self.hue_speed = 0.3  # degrees per frame
        self.time = 0.0
        self.dt = 0.005

        # timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(16)

    # -------------------------
    # Drawing & mouse events
    # -------------------------
    def toggle_drawing_mode(self):
        self.drawing_mode = self.draw_mode_btn_is_checked()
        if self.drawing_mode:
            self.start_new_stroke()

    def draw_mode_btn_is_checked(self):
        return self.draw_mode_btn_state() if hasattr(self, 'draw_mode_btn_state') else self.draw_mode_btn_state_fallback()

    def draw_mode_btn_state_fallback(self):
        # For compatibility with older PyQt6 button API
        return self.draw_mode_btn.isChecked()

    # simpler direct:
    def toggle_drawing_mode(self):
        self.drawing_mode = self.draw_mode_btn.isChecked()
        if self.drawing_mode:
            self.start_new_stroke()

    def start_new_stroke(self):
        if self.current_stroke:
            # finalize previous stroke
            self.finalize_current_stroke()
        self.current_stroke = []

    def mousePressEvent(self, event: QMouseEvent):
        if self.drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            self.current_stroke = []
            x = event.position().x() - self.width()/2
            y = event.position().y() - self.height()/2
            self.current_stroke.append(complex(x, y))
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing_mode and (event.buttons() & Qt.MouseButton.LeftButton):
            x = event.position().x() - self.width()/2
            y = event.position().y() - self.height()/2
            self.current_stroke.append(complex(x, y))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.drawing_mode and event.button() == Qt.MouseButton.LeftButton:
            # finalize
            self.finalize_current_stroke()

    def finalize_current_stroke(self):
        if not self.current_stroke:
            return
        # simplify/resample stroke: make evenly spaced sample count
        pts = np.array(self.current_stroke, dtype=complex)
        # remove duplicates
        if len(pts) < 2:
            return
        # resample to fixed number for stability
        L = max(64, min(2000, len(pts)))
        ts = np.linspace(0, 1, L)
        # cumulative length param
        d = np.sqrt(np.sum(np.diff(pts).real**2 + np.diff(pts).imag**2))
        # param by normalized cumulative distance:
        seg_lens = np.insert(np.cumsum(np.abs(np.diff(pts))), 0, 0.0)
        if seg_lens[-1] == 0:
            resampled = np.repeat(pts[0], L)
        else:
            seg_lens_norm = seg_lens/seg_lens[-1]
            resampled = np.array([np.interp(t, seg_lens_norm, pts.real) + 1j*np.interp(t, seg_lens_norm, pts.imag) for t in ts])
        self.strokes.append(resampled)
        self.trails.append([])  # one trail per stroke
        self.current_stroke = []
        self.recompute_all_coeffs()
        self.update()

    # -------------------------
    # Compute Fourier for strokes
    # -------------------------
    def recompute_all_coeffs(self):
        num_terms = self.term_slider.value()
        self.stroke_coeffs = []
        for stroke in self.strokes:
            coeffs = compute_fourier(stroke, num_terms)
            self.stroke_coeffs.append(coeffs)
        # combined
        if self.combine_checkbox.isChecked() and len(self.strokes) > 0:
            # concatenate strokes with small separators (zeros) and re-sample to uniform length
            all_pts = np.concatenate(self.strokes)
            # resample to fixed length
            N = max(256, len(all_pts))
            idx = np.linspace(0, len(all_pts)-1, N)
            combined = np.interp(idx, np.arange(len(all_pts)), all_pts.real) + 1j * np.interp(idx, np.arange(len(all_pts)), all_pts.imag)
            self.combined_coeffs = compute_fourier(combined, num_terms)
        else:
            self.combined_coeffs = []

    # -------------------------
    # SVG load/save
    # -------------------------
    def on_load_svg(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open SVG", "", "SVG Files (*.svg)")
        if not file:
            return
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            # Namespace handling: find all <path> nodes by tag ending with 'path'
            path_elems = []
            for elem in root.iter():
                tag = elem.tag
                if tag.lower().endswith('path'):
                    path_elems.append(elem)
            loaded = 0
            self.strokes = []
            self.trails = []
            for p in path_elems:
                d = p.attrib.get('d')
                if not d:
                    continue
                pts = parse_svg_path(d, num_samples=600)
                if pts.size == 0:
                    continue
                # normalize & center
                pts = np.array(pts, dtype=complex)
                pts -= pts.mean()
                maxabs = np.max(np.abs(pts))
                if maxabs != 0:
                    pts /= maxabs
                # scale to widget size
                scale = min(self.width(), self.height())*0.35
                pts = pts * scale
                self.strokes.append(pts)
                self.trails.append([])
                loaded += 1
            if loaded:
                self.recompute_all_coeffs()
                self.update()
        except Exception as e:
            print("SVG load failed:", e)

    def on_save_svg(self):
        if not self.strokes and not self.trails:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save SVG", "", "SVG Files (*.svg)")
        if not filename:
            return
        # convert trails/strokes into stroke lists in canvas coordinates (centered)
        strokes_for_svg = []
        for s in self.strokes:
            # shift back to positive coords for SVG
            pts = [(p.real + self.width()/2, p.imag + self.height()/2) for p in s]
            strokes_for_svg.append([complex(x,y) for x,y in pts])
        save_multistroke_svg(strokes_for_svg, filename, canvas_size=(self.width(), self.height()))

    # -------------------------
    # Clear
    # -------------------------
    def clear_strokes(self):
        self.strokes = []
        self.stroke_coeffs = []
        self.combined_coeffs = []
        self.trails = []
        self.current_stroke = []
        self.update()

    # -------------------------
    # Play/pause
    # -------------------------
    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.setText("Play" if not self.playing else "Pause")

    # -------------------------
    # Animation tick
    # -------------------------
    def update_animation(self):
        if not self.playing:
            return
        self.time += self.dt
        self.hue_offset = (self.hue_offset + self.hue_speed) % 360

        # compute current tip(s) and append to trails
        if self.combine_checkbox.isChecked() and self.combined_coeffs:
            tip = 0+0j
            for k, c in self.combined_coeffs[:self.term_slider.value()]:
                tip += c * np.exp(2j * np.pi * k * self.time)
            self.trails = [[tip]]  # single trail for combined
        else:
            # per-stroke
            for si, coeffs in enumerate(self.stroke_coeffs):
                tip = 0+0j
                for k, c in coeffs[:self.term_slider.value()]:
                    tip += c * np.exp(2j * np.pi * k * self.time)
                # center tip in widget coords
                # (we used centered coords for strokes)
                self.trails[si].append(tip)
                if len(self.trails[si]) > self.max_trail:
                    self.trails[si].pop(0)

        self.update()

    # -------------------------
    # Painting
    # -------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        cx, cy = self.width()/2, self.height()/2

        # Draw original strokes faintly
        pen = QPen(QColor(80, 80, 80, 180))
        pen.setWidth(1)
        painter.setPen(pen)
        for s in self.strokes:
            if len(s) < 2:
                continue
            prev = (cx + s[0].real, cy + s[0].imag)
            for p in s[1:]:
                cur = (cx + p.real, cy + p.imag)
                painter.drawLine(int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]))
                prev = cur

        # Draw epicycles & tips
        if self.combine_checkbox.isChecked() and self.combined_coeffs:
            # single chain
            x, y = 0.0, 0.0
            coeffs = self.combined_coeffs
            for idx, (k,c) in enumerate(coeffs[:self.term_slider.value()]):
                prevx, prevy = x, y
                freq = int(k)
                radius = abs(c) * min(self.width(), self.height()) * 0.5
                phase = np.angle(c)
                x += radius * math.cos(2*math.pi*freq*self.time + phase)
                y += radius * math.sin(2*math.pi*freq*self.time + phase)
                # color by index/hue
                hue = int((self.hue_offset + idx*360/len(coeffs)) % 360)
                color = QColor.fromHsv(hue, 200, 255, 140)
                pen = QPen(color)
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawEllipse(int(cx+prevx-radius), int(cy+prevy-radius), int(2*radius), int(2*radius))
                # radius line
                pen = QPen(QColor(255,255,255,200))
                painter.setPen(pen)
                painter.drawLine(int(cx+prevx), int(cy+prevy), int(cx+x), int(cy+y))
            # draw trail
            if self.trails:
                trail = self.trails[0]
                for i in range(1, len(trail)):
                    p1, p2 = trail[i-1], trail[i]
                    hue = int((self.hue_offset + i*120/len(trail)) % 360)
                    col = QColor.fromHsv(hue, 255, 255, int(255 * i / len(trail)))
                    pen = QPen(col)
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.drawLine(int(cx + p1.real), int(cy + p1.imag), int(cx + p2.real), int(cy + p2.imag))
        else:
            # per stroke chains
            for si, coeffs in enumerate(self.stroke_coeffs):
                # compute epicycles chain starting at center
                x, y = 0.0, 0.0
                for idx, (k,c) in enumerate(coeffs[:self.term_slider.value()]):
                    prevx, prevy = x, y
                    freq = int(k)
                    radius = abs(c) * min(self.width(), self.height()) * 0.5
                    phase = np.angle(c)
                    x += radius * math.cos(2*math.pi*freq*self.time + phase)
                    y += radius * math.sin(2*math.pi*freq*self.time + phase)
                    hue = int((self.hue_offset + idx*360/len(coeffs)) % 360)
                    color = QColor.fromHsv(hue, 200, 255, 120)
                    pen = QPen(color)
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawEllipse(int(cx+prevx-radius), int(cy+prevy-radius), int(2*radius), int(2*radius))
                    # radius line
                    pen = QPen(QColor(255,255,255,200))
                    painter.setPen(pen)
                    painter.drawLine(int(cx+prevx), int(cy+prevy), int(cx+x), int(cy+y))
                # draw trail for this stroke
                if si < len(self.trails):
                    trail = self.trails[si]
                    for i in range(1, len(trail)):
                        p1, p2 = trail[i-1], trail[i]
                        hue = int((self.hue_offset + i*120/len(trail)) % 360)
                        col = QColor.fromHsv(hue, 255, 255, int(200 * i / len(trail)))
                        pen = QPen(col)
                        pen.setWidth(2)
                        painter.setPen(pen)
                        painter.drawLine(int(cx + p1.real), int(cy + p1.imag), int(cx + p2.real), int(cy + p2.imag))

        # optionally draw active current stroke being drawn
        if self.current_stroke:
            pen = QPen(QColor(255, 255, 255, 220))
            pen.setWidth(2)
            painter.setPen(pen)
            prev = (cx + self.current_stroke[0].real, cy + self.current_stroke[0].imag)
            for p in self.current_stroke[1:]:
                cur = (cx + p.real, cy + p.imag)
                painter.drawLine(int(prev[0]), int(prev[1]), int(cur[0]), int(cur[1]))
                prev = cur

    # -------------------------
    # Boilerplate: start app
    # -------------------------
def main():
    # High DPI attributes must be set before creating QApplication
    app = QApplication(sys.argv)
    w = FourierWidget()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
