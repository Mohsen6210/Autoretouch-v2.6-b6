import os
import sys
import threading
import traceback
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk, ImageOps

import engine as E

Image.MAX_IMAGE_PIXELS = None
LOG_PATH = os.path.join(tempfile.gettempdir(), "AutoRetouch_error.log")
PREVIEW_MAX = 2000
DB_RANGE = 14.0  # L units reached at full dodge/burn strength
ANALYSIS_MAX = 2400
ACTIVE = "#1f6aa5"
IDLE = "#3a3a3a"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def log_error(text):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def load_image(path):
    pil = Image.open(path)
    icc = pil.info.get("icc_profile")
    pil = ImageOps.exif_transpose(pil)  # applies the camera rotation
    exif = None
    try:
        ex = pil.getexif()
        if ex:
            ex[0x0112] = 1
            exif = ex.tobytes()
    except Exception:
        exif = None
    pil = pil.convert("RGB")
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr, icc, exif


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Auto Retouch - Smart Dodge & Burn")
        self.geometry("1400x860")
        self.minsize(900, 600)

        # image state
        self.path = None
        self.full = None
        self.icc = None
        self.exif = None
        self.A = None
        self.an = None
        self.P = None
        self.P_rgb = None
        self.lab_P = None
        self.fp_P = 100.0
        self.mask = None
        self.mask_eff = None
        self.maps = None
        self.bbox = None
        self.result_rgb = None
        self.disp = None
        self.undo = []
        self.D = None
        self.db_bbox = None
        self.mode = "auto"
        self.view = "preview"
        self.tool = "move"

        # view state
        self.zoom = 1.0
        self.tx = 0.0
        self.ty = 0.0
        self.auto_fit = True
        self._photo = None
        self._pan_last = None
        self._last_pt = None
        self._painting = False
        self.space = False

        # background work
        self._results = []
        self._rc_running = False
        self._rc_pending = False
        self._rc_after = None
        self._comp_after = None
        self._busy = False

        self._build_ui()
        self._bind_keys()
        self.after(80, self._poll)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 3))
        ctk.CTkButton(top, text="Open", width=70, command=self.open_dialog).pack(side="left", padx=(6, 10), pady=6)
        self.btn_view = {}
        for key, text in (("original", "Original"), ("preview", "Preview"), ("mask", "Mask")):
            b = ctk.CTkButton(top, text=text, width=80, fg_color=IDLE, command=lambda k=key: self.set_view(k))
            b.pack(side="left", padx=2)
            self.btn_view[key] = b

        ctk.CTkButton(top, text="Reset (R)", width=80, fg_color=IDLE, command=self.reset_sliders).pack(side="left", padx=(16, 0))

        sf = ctk.CTkFrame(self)
        sf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=3)
        self.sl = {}
        for key, text, lo, hi, init in (("warmth", "Warmth", -50, 50, 0),
                                        ("blend", "Blend", 0, 100, 100),
                                        ("strength", "Strength", 0, 200, 100),
                                        ("texture", "Texture", 0, 200, 100)):
            f = ctk.CTkFrame(sf, fg_color="transparent")
            f.pack(side="left", padx=(4, 0), pady=4)
            lab = ctk.CTkLabel(f, text=f"{text}: {init}", width=92, anchor="w", font=ctk.CTkFont(size=12))
            lab.pack(side="left")
            s = ctk.CTkSlider(f, from_=lo, to=hi, number_of_steps=hi - lo, width=88,
                              command=lambda v, k=key: self._on_slider(k))
            s.set(init)
            s.pack(side="left")
            self.sl[key] = (s, lab, text)

        left = ctk.CTkScrollableFrame(self, width=108)
        left.grid(row=2, column=0, sticky="ns", padx=(6, 3), pady=3)
        self.btn_tool = {}
        for key, text in (("move", "Move (V)"), ("brush", "Brush (B)"), ("eraser", "Eraser (E)"),
                          ("dodge", "Dodge (1)"), ("burn", "Burn (2)")):
            b = ctk.CTkButton(left, text=text, width=80, fg_color=IDLE, command=lambda k=key: self.set_tool(k))
            b.pack(padx=5, pady=(6, 2))
            self.btn_tool[key] = b
        ctk.CTkButton(left, text="Invert (I)", width=80, fg_color=IDLE, command=self.invert_mask).pack(padx=5, pady=2)
        ctk.CTkButton(left, text="Undo (Ctrl+Z)", width=80, fg_color=IDLE, command=self.undo_mask).pack(padx=5, pady=2)
        ctk.CTkButton(left, text="Re-detect", width=80, fg_color=IDLE, command=lambda: self.set_mode(self.mode)).pack(padx=5, pady=2)
        ctk.CTkButton(left, text="Clear D&B", width=80, fg_color=IDLE, command=self.clear_db).pack(padx=5, pady=2)
        def side_slider(text, lo, hi, init, cb=None):
            lab = ctk.CTkLabel(left, text=f"{text}: {init}")
            lab.pack(pady=(10, 0))
            sl = ctk.CTkSlider(left, from_=lo, to=hi, number_of_steps=hi - lo, width=80,
                               command=lambda v: (lab.configure(text=f"{text}: {int(v)}"), cb() if cb else None))
            sl.set(init)
            sl.pack(padx=5, pady=2)
            return sl

        self.brush = side_slider("Brush size", 3, 200, 30)
        self.hard = side_slider("Hardness", 0, 100, 40)
        self.opac = side_slider("Opacity / Flow", 5, 100, 100)
        self.fth = side_slider("Mask feather px", 0, 80, 20, cb=self._feather_changed)
        self.mlight = side_slider("Mask light", -100, 100, 0, cb=self._db_changed)
        self.dbamt = side_slider("D&B amount", 0, 200, 100, cb=self._db_changed)
        self.db_limit = ctk.CTkCheckBox(left, text="D&B on skin\nmask only", command=self._db_changed, font=ctk.CTkFont(size=11))
        self.db_limit.pack(padx=4, pady=(10, 2), anchor="w")
        ctk.CTkLabel(left, text="Dodge / Burn paint light\non the photo like\nPhotoshop (flow = Opacity).\nCtrl+click (Brush/Eraser):\nfill / clear mask\n\nWheel: zoom\nSpace+drag: pan\n[ ]: brush size\nCtrl+1: 100%",
                     justify="left", font=ctk.CTkFont(size=11), text_color="#9a9a9a").pack(padx=4, pady=12)

        right = ctk.CTkScrollableFrame(self, width=250, label_text="Adjust (whole image)")
        right.grid(row=2, column=2, sticky="ns", padx=(3, 6), pady=3)
        self.adj_limit = ctk.CTkCheckBox(right, text="Only on skin mask", command=self._composite, font=ctk.CTkFont(size=11))
        self.adj_limit.pack(anchor="w", padx=4, pady=(2, 2))
        ctk.CTkButton(right, text="Reset adjustments", fg_color=IDLE, command=self.reset_adjust).pack(padx=4, pady=(2, 4))
        self.adj, self.adj_var, self.adj_sec, self.sec_on = {}, {}, {}, {}
        blue_yellow = [(45, 60, 200), (120, 120, 120), (232, 220, 40)]
        green_magenta = [(40, 170, 60), (120, 120, 120), (190, 40, 170)]
        import colorsys
        rainbow = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(h / 6.0, 0.8, 0.95)) for h in range(7)]
        sections = (
            ("Light", (("brightness", "Exposure", None), ("contrast", "Contrast", None),
                       ("highlights", "Highlights", None), ("shadows", "Shadows", None),
                       ("whites", "Whites", None), ("blacks", "Blacks", None))),
            ("Color", (("temp", "Temperature", blue_yellow), ("tint", "Tint", green_magenta),
                       ("hue", "Hue", rainbow), ("vib", "Vibrance", None), ("sat", "Saturation", None),
                       ("red", "Red", None), ("green", "Green", None), ("blue", "Blue", None))),
            ("Detail", (("sharpen", "Sharpen", None), ("blur", "Blur", None))),
        )
        for title, items in sections:
            body = self._make_section(right, title)
            for key, text, grad in items:
                lo = 0 if key == "blur" else -100
                self._make_adj_row(body, key, text, lo, 100, grad, title)

        self.canvas = tk.Canvas(self, bg="#1c1c1c", highlightthickness=0, cursor="fleur")
        self.canvas.grid(row=2, column=1, sticky="nsew", pady=3)

        bottom = ctk.CTkFrame(self)
        bottom.grid(row=3, column=0, columnspan=3, sticky="ew", padx=6, pady=3)
        self.btn_mode = {}
        for key, text in (("auto", "Auto"), ("face", "Face"), ("half", "Half-body"), ("full", "Full-body")):
            b = ctk.CTkButton(bottom, text=text, width=84, fg_color=IDLE, command=lambda k=key: self.set_mode(k))
            b.pack(side="left", padx=2, pady=6)
            self.btn_mode[key] = b
        self.btn_mode["auto"].configure(fg_color=ACTIVE)

        zf = ctk.CTkFrame(bottom, fg_color="transparent")
        zf.pack(side="left", expand=True)
        ctk.CTkButton(zf, text="-", width=36, command=lambda: self.zoom_by(1 / 1.25)).pack(side="left", padx=2)
        self.zoom_lab = ctk.CTkLabel(zf, text="100%", width=60)
        self.zoom_lab.pack(side="left")
        ctk.CTkButton(zf, text="+", width=36, command=lambda: self.zoom_by(1.25)).pack(side="left", padx=2)
        ctk.CTkButton(zf, text="Fit", width=46, fg_color=IDLE, command=self.fit).pack(side="left", padx=(10, 2))
        ctk.CTkButton(zf, text="100%", width=50, fg_color=IDLE, command=lambda: self.set_zoom(1.0)).pack(side="left", padx=2)

        ctk.CTkButton(bottom, text="Close", width=70, fg_color=IDLE, command=self.destroy).pack(side="right", padx=(2, 6))
        ctk.CTkButton(bottom, text="Save As", width=80, command=lambda: self.save(True)).pack(side="right", padx=2)
        self.btn_save = ctk.CTkButton(bottom, text="Save", width=80, command=lambda: self.save(False))
        self.btn_save.pack(side="right", padx=2)

        self.status = ctk.CTkLabel(self, text="Open a portrait to start.", anchor="w")
        self.status.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 5))

        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        for b in ("2", "3"):
            self.canvas.bind(f"<ButtonPress-{b}>", self._pan_start)
            self.canvas.bind(f"<B{b}-Motion>", self._pan_move)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel_zoom(e, 1.15))
        self.canvas.bind("<Button-5>", lambda e: self._wheel_zoom(e, 1 / 1.15))
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", lambda e: self.canvas.delete("cur"))
        self._refresh_buttons()

    def _bind_keys(self):
        self.bind("<Control-z>", lambda e: self.undo_mask())
        self.bind_all("<KeyPress-space>", self._space_down)
        self.bind_all("<KeyRelease-space>", self._space_up)
        self.bind("<KeyPress-r>", lambda e: self.reset_sliders())
        self.bind("<Control-Key-1>", lambda e: self.set_zoom(1.0))
        self.bind("<KeyPress-bracketleft>", lambda e: self._size_step(0.85))
        self.bind("<KeyPress-bracketright>", lambda e: self._size_step(1.18))
        for k, f in (("v", lambda: self.set_tool("move")), ("b", lambda: self.set_tool("brush")),
                     ("e", lambda: self.set_tool("eraser")), ("i", self.invert_mask),
                     ("0", self.fit), ("1", lambda: self.set_tool("dodge")), ("2", lambda: self.set_tool("burn")),
                     ("plus", lambda: self.zoom_by(1.25)), ("equal", lambda: self.zoom_by(1.25)),
                     ("minus", lambda: self.zoom_by(1 / 1.25))):
            self.bind(f"<KeyPress-{k}>", lambda e, f=f: f())

    def _space_down(self, e):
        self.space = True
        self.canvas.configure(cursor="fleur")
        return "break"

    def _space_up(self, e):
        self.space = False
        self._refresh_buttons()
        return "break"

    def _size_step(self, f):
        v = float(np.clip(self.brush.get() * f, 3, 200))
        self.brush.set(v)

    def reset_sliders(self):
        for k, v in (("warmth", 0), ("blend", 100), ("strength", 100), ("texture", 100)):
            self.sl[k][0].set(v)
            self.sl[k][1].configure(text=f"{self.sl[k][2]}: {v}")
        self._composite()

    def _make_section(self, parent, title):
        hdr = ctk.CTkFrame(parent, fg_color="#3a3a3a", corner_radius=4)
        hdr.pack(fill="x", pady=(8, 0))
        body = ctk.CTkFrame(parent, fg_color="transparent")
        body.pack(fill="x")
        self.sec_on[title] = True
        state = {"open": True}

        def toggle_open():
            state["open"] = not state["open"]
            chev.configure(text=("v  " if state["open"] else ">  ") + title)
            if state["open"]:
                body.pack(fill="x", after=hdr)
            else:
                body.pack_forget()

        def toggle_eye():
            self.sec_on[title] = not self.sec_on[title]
            eye.configure(text="(o)" if self.sec_on[title] else "( )")
            self._composite()

        chev = ctk.CTkButton(hdr, text="v  " + title, anchor="w", width=170, height=26, fg_color="transparent",
                             hover_color="#4a4a4a", font=ctk.CTkFont(size=13, weight="bold"), command=toggle_open)
        chev.pack(side="left")
        eye = ctk.CTkButton(hdr, text="(o)", width=40, height=26, fg_color="transparent", hover_color="#4a4a4a",
                            command=toggle_eye)
        eye.pack(side="right")
        return body

    def _make_adj_row(self, parent, key, text, lo, hi, grad, section):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=4, pady=(5, 0))
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.pack(fill="x")
        lab = ctk.CTkLabel(top, text=text, anchor="w", font=ctk.CTkFont(size=12))
        lab.pack(side="left")
        var = tk.StringVar(value="0")
        ent = ctk.CTkEntry(top, width=54, height=24, textvariable=var, justify="center", font=ctk.CTkFont(size=12))
        ent.pack(side="right")
        inner = ent._entry  # typing must not trigger the app shortcuts
        inner.bindtags((str(inner), "Entry"))
        if grad:
            strip = tk.Canvas(row, height=5, width=200, highlightthickness=0, bg="#2b2b2b")
            strip.pack(fill="x", pady=(3, 0))
            def draw(e=None, strip=strip, grad=grad):
                w = max(40, strip.winfo_width())
                strip.delete("all")
                n = 60
                for i in range(n):
                    t = i / (n - 1) * (len(grad) - 1)
                    a, b = int(t), min(int(t) + 1, len(grad) - 1)
                    f = t - a
                    c = tuple(int(grad[a][k] + (grad[b][k] - grad[a][k]) * f) for k in range(3))
                    strip.create_rectangle(i * w / n, 0, (i + 1) * w / n + 1, 6, fill="#%02x%02x%02x" % c, outline="")

            strip.bind("<Configure>", draw)
        sl = ctk.CTkSlider(row, from_=lo, to=hi, number_of_steps=hi - lo, width=200,
                           command=lambda v, k=key: self._adj_changed(k))
        sl.set(0)
        sl.pack(fill="x", pady=(1, 2))
        ent.bind("<Return>", lambda e, k=key: self._adj_entry(k))
        ent.bind("<FocusOut>", lambda e, k=key: self._adj_entry(k))
        lab.bind("<Double-Button-1>", lambda e, k=key: self._adj_set(k, 0))
        self.adj[key], self.adj_var[key], self.adj_sec[key] = (sl, lab, text), var, section

    def _adj_set(self, key, v):
        sl = self.adj[key][0]
        v = int(max(sl.cget("from_"), min(sl.cget("to"), v)))
        sl.set(v)
        self.adj_var[key].set(str(v))
        self._adj_changed(key)

    def _adj_entry(self, key):
        try:
            v = int(round(float(self.adj_var[key].get().replace(",", "."))))
        except ValueError:
            v = int(self.adj[key][0].get())
        self._adj_set(key, v)
        self.canvas.focus_set()

    def _adj_changed(self, key):
        sl, lab, text = self.adj[key]
        self.adj_var[key].set(str(int(sl.get())))
        if self._comp_after:
            self.after_cancel(self._comp_after)
        self._comp_after = self.after(50, self._composite)

    def reset_adjust(self):
        for k, (sl, lab, text) in self.adj.items():
            sl.set(0)
            self.adj_var[k].set("0")
        self._composite()

    def _feather_changed(self):
        if self.mask is not None:
            self._mask_changed()

    def report_callback_exception(self, exc, val, tb):
        err = "".join(traceback.format_exception(exc, val, tb))
        log_error(err)
        messagebox.showerror("Unexpected error", err[-900:])

    def _refresh_buttons(self):
        for k, b in self.btn_view.items():
            b.configure(fg_color=ACTIVE if k == self.view else IDLE)
        for k, b in self.btn_tool.items():
            b.configure(fg_color=ACTIVE if k == self.tool else IDLE)
        for k, b in self.btn_mode.items():
            b.configure(fg_color=ACTIVE if k == self.mode else IDLE)
        self.canvas.configure(cursor="fleur" if self.tool == "move" else "crosshair")

    def set_status(self, t):
        self.status.configure(text=t)

    # ------------------------------------------------------------ opening
    def open_dialog(self):
        if self._busy:
            return
        p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp")])
        if p:
            self.open_path(p)

    def open_path(self, p):
        try:
            self.set_status("Loading image...")
            self.update_idletasks()
            full, icc, exif = load_image(p)
        except Exception:
            err = traceback.format_exc()
            log_error(err)
            messagebox.showerror("Cannot open image", err[-700:])
            return
        self.path, self.full, self.icc, self.exif = p, full, icc, exif
        self.P = E.resize_max(full, PREVIEW_MAX)
        self.P_rgb = cv2.cvtColor(self.P, cv2.COLOR_BGR2RGB)
        self.lab_P = cv2.cvtColor(self.P.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        self.A = E.resize_max(full, ANALYSIS_MAX)
        self.an = None
        self.mask = np.zeros(self.P.shape[:2], np.float32)
        self.mask_eff = self.mask.copy()
        self.D = np.zeros(self.P.shape[:2], np.float32)
        self.db_bbox = None
        self.maps = None
        self.bbox = None
        self.result_rgb = None
        self.undo = []
        self.auto_fit = True
        self.view = "original"
        self._refresh_buttons()
        self._update_disp()
        self.fit()
        self.set_status("Detecting faces... (first run takes a few seconds)")
        A, mode = self.A, self.mode

        def work():
            try:
                an = E.analyze(A)
                m, info = E.build_mask(an, mode)
                self._results.append(("analysis", an, m, info))
            except Exception:
                self._results.append(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def set_mode(self, mode):
        self.mode = mode
        self._refresh_buttons()
        if self.an is None or self._busy:
            return
        self.set_status("Rebuilding mask...")
        an = self.an

        def work():
            try:
                m, info = E.build_mask(an, mode)
                self._results.append(("mode", m, info))
            except Exception:
                self._results.append(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    # --------------------------------------------------------- bg results
    def _poll(self):
        while self._results:
            r = self._results.pop(0)
            try:
                self._handle(r)
            except Exception:
                err = traceback.format_exc()
                log_error(err)
                messagebox.showerror("Error", err[-800:])
        self.after(80, self._poll)

    def _handle(self, r):
        kind = r[0]
        if kind == "analysis":
            self.an = r[1]
            self._apply_mask_result(r[2], r[3], first=True)
        elif kind == "mode":
            self._apply_mask_result(r[1], r[2], first=False)
        elif kind == "maps":
            self.maps = r[1]
            ys, xs = np.where(self.maps[..., 0] > 0.002)
            self.bbox = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1) if len(ys) else None
            self._rc_running = False
            self._composite()
            if self._rc_pending:
                self._rc_pending = False
                self._start_recompute()
        elif kind == "progress":
            self.set_status(f"Saving... {int(r[1] * 100)}%")
        elif kind == "saved":
            self._busy = False
            self.btn_save.configure(state="normal")
            self.set_status(f"Saved: {r[1]}")
            messagebox.showinfo("Saved", f"Saved to:\n{r[1]}")
        elif kind == "error":
            self._busy = False
            self._rc_running = False
            self.btn_save.configure(state="normal")
            log_error(r[1])
            self.set_status("An error occurred")
            messagebox.showerror("Error", r[1][-900:])

    def _apply_mask_result(self, mask_A, info, first):
        Ph, Pw = self.P.shape[:2]
        self.mask = cv2.resize(mask_A, (Pw, Ph), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self.an is not None and self.an.faces:
            self.fp_P = self.an.med_face_px * Ph / self.A.shape[0]
        else:
            self.fp_P = 0.25 * Ph
        self.undo = []
        names = {"face": "close-up", "half": "half-body", "full": "full-body"}
        if info["faces"] == 0:
            self.set_status("No face found. Paint skin with the Brush, or press Invert to select everything.")
        else:
            warn = "" if info["mesh"] == info["faces"] else f" ({info['faces'] - info['mesh']} approximate)"
            self.set_status(f"{info['faces']} face(s){warn} - mode: {names.get(info['mode'], info['mode'])}. "
                            f"Fix the mask with Brush / Eraser (see the Mask view).")
        if first:
            self.view = "preview"
            self._refresh_buttons()
        self._mask_changed()

    # ------------------------------------------------------ mask editing
    def _mask_changed(self):
        self.mask_eff = E.feather(self.mask, self.fth.get(), max(self.mask.shape))
        self._schedule_recompute()
        self._update_disp()
        self._render()

    def _schedule_recompute(self, delay=150):
        if self._rc_after:
            self.after_cancel(self._rc_after)
        self._rc_after = self.after(delay, self._start_recompute)

    def _start_recompute(self):
        self._rc_after = None
        if self.P is None:
            return
        if self._rc_running:
            self._rc_pending = True
            return
        self._rc_running = True
        me, lab, fp = self.mask_eff.copy(), self.lab_P, self.fp_P

        def work():
            try:
                dL, da, db = E.compute_delta(lab, me, fp)
                self._results.append(("maps", np.dstack([me, dL, da, db]).astype(np.float32)))
            except Exception:
                self._results.append(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    def push_undo(self, kind="mask"):
        if kind == "db":
            self.undo.append(("db", np.round(np.clip(self.D, -1, 1) * 127).astype(np.int8)))
        else:
            self.undo.append(("mask", (np.clip(self.mask, 0, 1) * 255).astype(np.uint8)))
        if len(self.undo) > 15:
            self.undo.pop(0)

    def undo_mask(self):
        if not self.undo or self.mask is None:
            return
        kind, arr = self.undo.pop()
        if kind == "db":
            self.D = arr.astype(np.float32) / 127.0
            self.db_bbox = (0, self.D.shape[0], 0, self.D.shape[1]) if np.any(arr) else None
            self._db_changed()
        else:
            self.mask = arr.astype(np.float32) / 255.0
            self._mask_changed()

    def clear_db(self):
        if self.D is None or self.db_bbox is None:
            return
        self.push_undo("db")
        self.D[:] = 0
        self.db_bbox = None
        self._db_changed()

    def _db_changed(self):
        if self._comp_after:
            self.after_cancel(self._comp_after)
        self._comp_after = self.after(40, self._composite)

    def invert_mask(self):
        if self.mask is None:
            return
        self.push_undo()
        self.mask = 1.0 - self.mask
        self._mask_changed()

    # ------------------------------------------------------ view / render
    def set_view(self, v):
        self.view = v
        self._refresh_buttons()
        self._update_disp()
        self._render()

    def set_tool(self, t):
        self.tool = t
        if t != "move" and self.view == "original":
            self.view = "preview"
        self._refresh_buttons()
        self._update_disp()
        self._render()

    def _on_slider(self, key):
        s, lab, text = self.sl[key]
        lab.configure(text=f"{text}: {int(s.get())}")
        if self._comp_after:
            self.after_cancel(self._comp_after)
        self._comp_after = self.after(50, self._composite)

    def _params(self):
        v = {k: self.sl[k][0].get() for k in self.sl}
        p = dict(blend=v["blend"] / 100.0, strength=v["strength"] / 100.0, warm=v["warmth"],
                 texture=v["texture"] / 100.0, mask_light=self.mlight.get() / 100.0,
                 dbamt=self.dbamt.get() / 100.0, db_limit=bool(self.db_limit.get()),
                 adj_limit=bool(self.adj_limit.get()))
        for k, (sl, _l, _t) in self.adj.items():
            p[k] = sl.get() / 100.0 if self.sec_on.get(self.adj_sec[k], True) else 0.0
        return p

    def _scales(self, p):
        long_p = float(max(self.P.shape[:2]))
        return (max(0.8, self.fp_P / 260.0), max(0.6, long_p / 1800.0), p["blur"] * long_p / 80.0)

    def _region(self, p):
        Ph, Pw = self.P.shape[:2]
        if E.adj_active(p) and not p["adj_limit"]:
            return (0, Ph, 0, Pw)
        regs = []
        if self.maps is not None and self.bbox is not None:
            y0, y1, x0, x1 = self.bbox
            regs.append((max(0, y0 - 12), min(Ph, y1 + 12), max(0, x0 - 12), min(Pw, x1 + 12)))
        if self.db_bbox is not None and p["dbamt"] > 0:
            regs.append(self.db_bbox)
        if not regs:
            return None
        return (min(r[0] for r in regs), max(r[1] for r in regs), min(r[2] for r in regs), max(r[3] for r in regs))

    def _composite(self):
        self._comp_after = None
        if self.P is None:
            return
        out = self.P
        p = self._params()
        reg = self._region(p)
        if reg is not None:
            y0, y1, x0, x1 = reg
            sub = (slice(y0, y1), slice(x0, x1))
            maps = self.maps[sub] if self.maps is not None else np.zeros((y1 - y0, x1 - x0, 4), np.float32)
            db = self.D[sub] * (DB_RANGE * p["dbamt"]) if self.db_bbox is not None else None
            out = self.P.copy()
            out[sub] = E.apply_maps(self.P[sub], maps, p, db=db, s=self._scales(p))
        self.result_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        self._update_disp()
        self._render()

    def _update_disp(self):
        if self.P_rgb is None:
            self.disp = None
        elif self.view == "original":
            self.disp = self.P_rgb
        elif self.view == "mask":
            m = np.clip(self.mask, 0, 1)[..., None] * 0.55
            red = np.array([255, 40, 40], np.float32)
            self.disp = (self.P_rgb * (1 - m) + red * m).astype(np.uint8)
        else:
            self.disp = self.result_rgb if self.result_rgb is not None else self.P_rgb

    def _render(self):
        c = self.canvas
        c.delete("img")
        if self.disp is None:
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        H, W = self.disp.shape[:2]
        z = self.zoom
        x0 = max(0, int(np.floor(-self.tx / z)))
        y0 = max(0, int(np.floor(-self.ty / z)))
        x1 = min(W, int(np.ceil((cw - self.tx) / z)))
        y1 = min(H, int(np.ceil((ch - self.ty) / z)))
        if x1 <= x0 or y1 <= y0:
            return
        crop = self.disp[y0:y1, x0:x1]
        dw, dh = max(1, int(round((x1 - x0) * z))), max(1, int(round((y1 - y0) * z)))
        interp = cv2.INTER_AREA if z < 1 else (cv2.INTER_NEAREST if z >= 4 else cv2.INTER_LINEAR)
        pic = cv2.resize(crop, (dw, dh), interpolation=interp)
        self._photo = ImageTk.PhotoImage(Image.fromarray(pic))
        c.create_image(self.tx + x0 * z, self.ty + y0 * z, anchor="nw", image=self._photo, tags="img")
        c.tag_lower("img")
        self.zoom_lab.configure(text=f"{int(round(z * 100))}%")

    def fit(self):
        if self.P is None:
            return
        cw, ch = max(50, self.canvas.winfo_width()), max(50, self.canvas.winfo_height())
        H, W = self.P.shape[:2]
        self.zoom = min(cw / W, ch / H) * 0.98
        self.tx = (cw - W * self.zoom) / 2
        self.ty = (ch - H * self.zoom) / 2
        self.auto_fit = True
        self._render()

    def set_zoom(self, z, cx=None, cy=None):
        if self.P is None:
            return
        z = float(np.clip(z, 0.05, 16))
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        cx = cw / 2 if cx is None else cx
        cy = ch / 2 if cy is None else cy
        ix, iy = (cx - self.tx) / self.zoom, (cy - self.ty) / self.zoom
        self.zoom = z
        self.tx, self.ty = cx - ix * z, cy - iy * z
        self.auto_fit = False
        self._render()

    def zoom_by(self, f):
        self.set_zoom(self.zoom * f)

    def _on_wheel(self, e):
        self._wheel_zoom(e, 1.15 ** (e.delta / 120.0))

    def _wheel_zoom(self, e, f):
        self.set_zoom(self.zoom * f, e.x, e.y)

    def _on_resize(self, e):
        if self.auto_fit and self.P is not None:
            self.fit()
        else:
            self._render()

    # -------------------------------------------------------- mouse tools
    def _to_img(self, x, y):
        return (x - self.tx) / self.zoom, (y - self.ty) / self.zoom

    def _pan_start(self, e):
        self._pan_last = (e.x, e.y)

    def _pan_move(self, e):
        if self._pan_last:
            self.tx += e.x - self._pan_last[0]
            self.ty += e.y - self._pan_last[1]
            self._pan_last = (e.x, e.y)
            self.auto_fit = False
            self._render()

    def _on_motion(self, e):
        self.canvas.delete("cur")
        if self.tool != "move" and self.P is not None:
            r = self.brush.get()
            col = "#ffb347" if self.tool == "burn" else "#ffffff"
            self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, outline=col, tags="cur")

    def _on_press(self, e):
        self.canvas.focus_set()
        if self.P is None:
            return
        if self.tool == "move" or self.space:
            self._pan_start(e)
            return
        if self.tool in ("dodge", "burn"):
            self.push_undo("db")
            self._painting = True
            x, y = self._to_img(e.x, e.y)
            self._last_pt = (x, y)
            self._paint_db(x, y, x, y)
            return
        self.push_undo()
        val = 1.0 if self.tool == "brush" else 0.0
        if e.state & 0x4:  # Ctrl+click: fill / clear
            self.mask[:] = val
            self._mask_changed()
            return
        self._painting = True
        x, y = self._to_img(e.x, e.y)
        self._last_pt = (x, y)
        self._paint(x, y, x, y, val)

    def _on_drag(self, e):
        if self.P is None:
            return
        if self.tool == "move" or self.space:
            self._pan_move(e)
            return
        self._on_motion(e)
        if self._painting:
            x, y = self._to_img(e.x, e.y)
            if self.tool in ("dodge", "burn"):
                self._paint_db(self._last_pt[0], self._last_pt[1], x, y)
            else:
                self._paint(self._last_pt[0], self._last_pt[1], x, y, 1.0 if self.tool == "brush" else 0.0)
            self._last_pt = (x, y)

    def _on_release(self, e):
        self._pan_last = None
        self._painting = False

    def _paint(self, x0, y0, x1, y1, val):
        H, W = self.mask.shape
        r = max(1.0, self.brush.get() / self.zoom)
        hard = self.hard.get() / 100.0
        op = self.opac.get() / 100.0
        dist = float(np.hypot(x1 - x0, y1 - y0))
        n = max(1, int(dist / max(1.0, r * 0.25)))
        for i in range(n + 1):
            t = i / n
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            xa, xb = int(max(0, x - r - 1)), int(min(W, x + r + 2))
            ya, yb = int(max(0, y - r - 1)), int(min(H, y + r + 2))
            if xb <= xa or yb <= ya:
                continue
            yy, xx = np.ogrid[ya:yb, xa:xb]
            d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            a = np.clip((r - d) / max(r * (1.0 - hard), 1e-3), 0, 1) * op
            roi = self.mask[ya:yb, xa:xb]
            if val > 0.5:
                np.maximum(roi, a.astype(np.float32), out=roi)
            else:
                roi *= (1.0 - a).astype(np.float32)
        self._mask_changed()

    def _paint_db(self, x0, y0, x1, y1):
        H, W = self.D.shape
        r = max(1.0, self.brush.get() / self.zoom)
        hard = self.hard.get() / 100.0
        flow = self.opac.get() / 100.0
        sign = 1.0 if self.tool == "dodge" else -1.0
        n = max(1, int(float(np.hypot(x1 - x0, y1 - y0)) / max(1.0, r * 0.25)))
        bb = self.db_bbox
        for i in range(n + 1):
            t = i / n
            x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
            xa, xb = int(max(0, x - r - 1)), int(min(W, x + r + 2))
            ya, yb = int(max(0, y - r - 1)), int(min(H, y + r + 2))
            if xb <= xa or yb <= ya:
                continue
            yy, xx = np.ogrid[ya:yb, xa:xb]
            d = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            a = np.clip((r - d) / max(r * (1.0 - hard), 1e-3), 0, 1) * flow * 0.07 * sign
            roi = self.D[ya:yb, xa:xb]
            roi += a.astype(np.float32)
            np.clip(roi, -1.0, 1.0, out=roi)
            bb = (ya, yb, xa, xb) if bb is None else (min(bb[0], ya), max(bb[1], yb), min(bb[2], xa), max(bb[3], xb))
        self.db_bbox = bb
        self._db_changed()

    # ------------------------------------------------------------- saving
    def save(self, ask):
        if self.full is None or self.P is None:
            messagebox.showinfo("Save", "Open an image first.")
            return
        if self._busy:
            return
        base, ext = os.path.splitext(self.path)
        if ask:
            p = filedialog.asksaveasfilename(
                defaultextension=".jpg", initialfile=os.path.basename(base) + "_retouched",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tif")])
            if not p:
                return
        else:
            e = ext.lower() if ext.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff") else ".png"
            p = base + "_retouched" + e
            n = 2
            while os.path.exists(p):
                p = f"{base}_retouched_{n}{e}"
                n += 1
        self._busy = True
        self.btn_save.configure(state="disabled")
        self.set_status("Saving... 0%")
        prm = self._params()
        db_p = (self.D * (DB_RANGE * prm["dbamt"])).astype(np.float32) if self.db_bbox is not None else None
        s_p = self._scales(prm)
        me, lab, fp, full = self.mask_eff.copy(), self.lab_P, self.fp_P, self.full
        icc, exif = self.icc, self.exif

        def work():
            try:
                dL, da, db = E.compute_delta(lab, me, fp)
                maps = np.dstack([me, dL, da, db]).astype(np.float32)
                out = E.render_full(full, maps, prm, db_p, s_p, progress=lambda f: self._results.append(("progress", f)))
                im = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                ext2 = os.path.splitext(p)[1].lower()
                kw = {}
                if icc:
                    kw["icc_profile"] = icc
                if exif:
                    kw["exif"] = exif
                if ext2 in (".jpg", ".jpeg"):
                    kw.update(quality=95, subsampling=0)
                elif ext2 in (".tif", ".tiff"):
                    kw.update(compression="tiff_lzw")
                    kw.pop("exif", None)
                try:
                    im.save(p, **kw)
                except Exception:
                    kw.pop("exif", None)
                    im.save(p, **kw)
                self._results.append(("saved", p))
            except Exception:
                self._results.append(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()


def selftest():
    """Used by the build pipeline: checks that the face / segmentation models load inside the exe."""
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
    out = os.path.join(base, "selftest.txt")
    try:
        import mediapipe as mp
        img = np.full((600, 800, 3), 170, np.uint8)
        an = E.analyze(img)
        E.build_mask(an, "auto")
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as fm:
            fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with open(out, "w") as f:
            f.write("OK")
    except Exception:
        with open(out, "w") as f:
            f.write(traceback.format_exc())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
        sys.exit(0)
    App().mainloop()
