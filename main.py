"""
main.py - PTFS ATC Assistant
Chat-style UI with flight strip board and AI context support.

Run:  python main.py
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import numpy as np

from audio import AudioCapture, AudioDevice, HAS_WASAPI
from llm import ATCResponder
from stt import Transcriber
import tts

# ── config ────────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Display label → model ID
_MODEL_OPTIONS: dict[str, str] = {
    "llama3.2:3b  (Fast)":          "llama3.2:3b",
    "gemma4:e4b  (Efficient)":      "gemma4:e4b",
    "llama3.1:8b  (Balanced)":      "llama3.1:8b",
    "llama3.2-vision:11b  (Vision)": "llama3.2-vision:11b",
}
_DEFAULT_MODEL_LABEL = "llama3.2:3b  (Fast)"


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(data: dict) -> None:
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _load_lexend() -> str:
    """Register Lexend font with GDI and return the family name."""
    font_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "sources", "Lexend-VariableFont_wght.ttf",
    )
    if os.path.exists(font_path) and os.name == "nt":
        try:
            ctypes.windll.gdi32.AddFontResourceW(font_path)
            return "Lexend"
        except Exception:
            pass
    return "Segoe UI"


# ── colour palette (light mode) ────────────────────────────────────────────────
BG_BASE       = "#f5f7fa"
BG_CARD       = "#ffffff"
BG_PANEL      = "#edf0f5"
PILOT_BG      = "#e8f0fe"
ATC_BG        = "#1967d2"
TEXT_DIM      = "#80868b"
TEXT_MAIN     = "#202124"
TEXT_ON_BLUE  = "#ffffff"
ACCENT        = "#1a73e8"
ACCENT_LIGHT  = "#e8f0fe"
GREEN         = "#188038"
ORANGE        = "#e37400"
RED           = "#d93025"
BORDER        = "#dadce0"

STATUS_COLORS = {
    "Clearance": "#7c3aed",
    "Pushback":  "#ea580c",
    "Taxi":      "#ca8a04",
    "Takeoff":   "#16a34a",
    "Departed":  "#6b7280",
    "Approach":  "#1a73e8",
    "Landing":   "#16a34a",
    "Landed":    "#6b7280",
}

DEPARTURE_STATUSES = ["Clearance", "Pushback", "Taxi", "Takeoff", "Departed"]
ARRIVAL_STATUSES   = ["Approach", "Landing", "Landed"]

# Font family — updated to "Lexend" in ATCAssistant.__init__ before any widgets are built
FONT_FAMILY = "Segoe UI"


# ── scrollable frame ────────────────────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent: tk.Widget, bg_color: str = BG_CARD, **kwargs) -> None:
        super().__init__(parent, bg=bg_color, **kwargs)

        self._canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0, bd=0)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self._canvas, bg=bg_color)
        self._win_id = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.inner.bind("<MouseWheel>", self._on_mousewheel)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def scroll_to_bottom(self) -> None:
        self.update_idletasks()
        self._canvas.yview_moveto(1.0)


# ── chat bubble ─────────────────────────────────────────────────────────────────
class ChatBubble(tk.Frame):
    """Single chat message bubble. is_pilot=True → left (light blue), False → right (dark blue)."""

    def __init__(self, parent: tk.Widget, text: str, is_pilot: bool = True) -> None:
        super().__init__(parent, bg=BG_CARD)
        self._text = text

        bubble_bg  = PILOT_BG if is_pilot else ATC_BG
        text_color = TEXT_MAIN if is_pilot else TEXT_ON_BLUE
        label_txt  = "Pilot" if is_pilot else "ATC Suggestion"
        anchor     = "w" if is_pilot else "e"

        row = tk.Frame(self, bg=BG_CARD)
        row.pack(fill="x", padx=12, pady=(4, 0))

        tk.Label(
            row, text=label_txt, bg=BG_CARD, fg=TEXT_DIM,
            font=(FONT_FAMILY, 8),
        ).pack(anchor=anchor)

        bubble = tk.Frame(row, bg=bubble_bg, padx=14, pady=9)
        bubble.pack(anchor=anchor)

        tk.Label(
            bubble,
            text=text,
            bg=bubble_bg,
            fg=text_color,
            font=(FONT_FAMILY, 11),
            wraplength=420,
            justify="left",
        ).pack(anchor="w")

        self._flash(bubble, bubble_bg)

        if not is_pilot:
            copy_lbl = tk.Label(
                row, text="⧈ copy", bg=BG_CARD, fg=TEXT_DIM,
                font=(FONT_FAMILY, 8), cursor="hand2",
            )
            copy_lbl.pack(anchor="e", pady=(2, 0))
            copy_lbl.bind("<Button-1>", self._copy)
            copy_lbl.bind("<Enter>", lambda e: copy_lbl.config(fg=ACCENT))
            copy_lbl.bind("<Leave>", lambda e: copy_lbl.config(fg=TEXT_DIM))

    def _flash(self, widget: tk.Widget, final_bg: str) -> None:
        """Brief highlight flash when a bubble appears."""
        flash = "#ffffff" if final_bg == PILOT_BG else "#4285f4"
        try:
            widget.config(bg=flash)
        except tk.TclError:
            return

        def _restore():
            try:
                widget.config(bg=final_bg)
            except tk.TclError:
                pass

        widget.after(150, _restore)

    def _copy(self, _event: tk.Event) -> None:
        root = self.winfo_toplevel()
        root.clipboard_clear()
        root.clipboard_append(self._text)


# ── flight strip ────────────────────────────────────────────────────────────────
class FlightStrip(tk.Frame):
    """One flight strip card (departure or arrival)."""

    def __init__(
        self,
        parent: tk.Widget,
        callsign: str,
        category: str,
        on_delete,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=BG_CARD, bd=0, **kwargs)
        self._on_delete = on_delete
        self._category  = category

        statuses = DEPARTURE_STATUSES if category == "departure" else ARRIVAL_STATUSES
        self._status_var = tk.StringVar(value=statuses[0])
        self._build(callsign, statuses)
        self._animate_in()

    def _build(self, callsign: str, statuses: list) -> None:
        initial_status = self._status_var.get()
        status_color   = STATUS_COLORS.get(initial_status, ACCENT)

        self._status_bar = tk.Frame(self, bg=status_color, width=6)
        self._status_bar.pack(side="left", fill="y")
        self._status_bar.pack_propagate(False)

        content = tk.Frame(self, bg=BG_CARD, padx=10, pady=8)
        content.pack(side="left", fill="both", expand=True)

        # Row 1 — callsign, status dropdown, delete
        row1 = tk.Frame(content, bg=BG_CARD)
        row1.pack(fill="x")

        self._callsign_var = tk.StringVar(value=callsign.upper())
        tk.Entry(
            row1, textvariable=self._callsign_var, bg=BG_CARD, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 12, "bold"), relief="flat", width=12,
            insertbackground=TEXT_MAIN,
        ).pack(side="left")

        status_cb = ttk.Combobox(
            row1, textvariable=self._status_var, values=statuses,
            width=11, state="readonly", font=(FONT_FAMILY, 9),
        )
        status_cb.pack(side="left", padx=(8, 0))
        status_cb.bind("<<ComboboxSelected>>", self._on_status_changed)

        del_lbl = tk.Label(
            row1, text="✕", bg=BG_CARD, fg=TEXT_DIM,
            font=(FONT_FAMILY, 10), cursor="hand2",
        )
        del_lbl.pack(side="right")
        del_lbl.bind("<Button-1>", lambda e: self._on_delete(self))
        del_lbl.bind("<Enter>", lambda e: del_lbl.config(fg=RED))
        del_lbl.bind("<Leave>", lambda e: del_lbl.config(fg=TEXT_DIM))

        # Row 2 — manual editable fields
        row2 = tk.Frame(content, bg=BG_CARD)
        row2.pack(fill="x", pady=(6, 0))

        self._aircraft_var = tk.StringVar()
        self._squawk_var   = tk.StringVar()
        self._alt_var      = tk.StringVar()
        self._notes_var    = tk.StringVar()

        for lbl, var, w in [
            ("Aircraft", self._aircraft_var, 8),
            ("Squawk",   self._squawk_var,   6),
            ("Alt",      self._alt_var,      7),
        ]:
            col = tk.Frame(row2, bg=BG_CARD)
            col.pack(side="left", padx=(0, 10))
            tk.Label(col, text=lbl, bg=BG_CARD, fg=TEXT_DIM,
                     font=(FONT_FAMILY, 7)).pack(anchor="w")
            tk.Entry(
                col, textvariable=var, bg=BG_PANEL, fg=TEXT_MAIN,
                font=(FONT_FAMILY, 9), relief="flat", width=w,
                insertbackground=TEXT_MAIN,
            ).pack()

        notes_col = tk.Frame(row2, bg=BG_CARD)
        notes_col.pack(side="left", fill="x", expand=True)
        tk.Label(notes_col, text="Notes", bg=BG_CARD, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 7)).pack(anchor="w")
        tk.Entry(
            notes_col, textvariable=self._notes_var, bg=BG_PANEL, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 9), relief="flat", insertbackground=TEXT_MAIN,
        ).pack(fill="x")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")

    def _on_status_changed(self, _event: tk.Event) -> None:
        color = STATUS_COLORS.get(self._status_var.get(), ACCENT)
        self._status_bar.config(bg=color)

    def _animate_in(self) -> None:
        """Flash the status bar from a highlight colour to its final colour on creation."""
        target = STATUS_COLORS.get(self._status_var.get(), ACCENT)
        try:
            self._status_bar.config(bg="#c8d8f0")
        except tk.TclError:
            return

        def _settle():
            try:
                self._status_bar.config(bg=target)
            except tk.TclError:
                pass

        self.after(220, _settle)


# ── flight strip board window ───────────────────────────────────────────────────
class FlightStripBoard(tk.Toplevel):
    """Standalone, independently minimisable flight strip board."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.title("Flight Strip Board — PTFS ATC")
        self.geometry("940x580")
        self.minsize(680, 360)
        self.configure(bg=BG_BASE)
        # No transient / grab_set → fully independent window

        self._dep_strips: List[FlightStrip] = []
        self._arr_strips: List[FlightStrip] = []
        self._build()

    def _build(self) -> None:
        header = tk.Frame(self, bg=BG_BASE, padx=16, pady=10)
        header.pack(fill="x")

        tk.Label(
            header, text="Flight Strip Board", bg=BG_BASE, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 14, "bold"),
        ).pack(side="left")

        tk.Button(
            header, text="+ Add Strip", bg=ACCENT, fg="white",
            font=(FONT_FAMILY, 9), relief="flat", padx=10, pady=4,
            activebackground="#1558b0", activeforeground="white",
            cursor="hand2", command=self._add_manual_strip,
        ).pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG_BASE)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # Departures column
        dep_frame = tk.Frame(body, bg=BG_BASE)
        dep_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)

        dep_hdr = tk.Frame(dep_frame, bg="#d4edda", padx=12, pady=7)
        dep_hdr.pack(fill="x")
        tk.Label(dep_hdr, text="▲  DEPARTURES", bg="#d4edda", fg="#155724",
                 font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        tk.Button(
            dep_hdr, text="+ Add", bg="#28a745", fg="white",
            font=(FONT_FAMILY, 8), relief="flat", padx=6, pady=2, cursor="hand2",
            command=lambda: self._add_strip("departure"),
        ).pack(side="right")

        self._dep_scroll = ScrollableFrame(dep_frame, bg_color=BG_BASE)
        self._dep_scroll.pack(fill="both", expand=True, pady=(4, 0))

        # Arrivals column
        arr_frame = tk.Frame(body, bg=BG_BASE)
        arr_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)

        arr_hdr = tk.Frame(arr_frame, bg="#cce5ff", padx=12, pady=7)
        arr_hdr.pack(fill="x")
        tk.Label(arr_hdr, text="▼  ARRIVALS", bg="#cce5ff", fg="#004085",
                 font=(FONT_FAMILY, 10, "bold")).pack(side="left")
        tk.Button(
            arr_hdr, text="+ Add", bg="#0069d9", fg="white",
            font=(FONT_FAMILY, 8), relief="flat", padx=6, pady=2, cursor="hand2",
            command=lambda: self._add_strip("arrival"),
        ).pack(side="right")

        self._arr_scroll = ScrollableFrame(arr_frame, bg_color=BG_BASE)
        self._arr_scroll.pack(fill="both", expand=True, pady=(4, 0))

    def _add_strip(self, category: str, callsign: str = "UNKNOWN") -> FlightStrip:
        strips = self._dep_strips if category == "departure" else self._arr_strips
        scroll = self._dep_scroll if category == "departure" else self._arr_scroll

        def on_delete(strip: FlightStrip) -> None:
            strip.destroy()
            if strip in strips:
                strips.remove(strip)

        strip = FlightStrip(
            scroll.inner,
            callsign=callsign,
            category=category,
            on_delete=on_delete,
        )
        strip.pack(fill="x", pady=(0, 6))
        strips.append(strip)
        scroll.scroll_to_bottom()
        return strip

    def _add_manual_strip(self) -> None:
        win = tk.Toplevel(self)
        win.title("Add Flight Strip")
        win.geometry("280x170")
        win.configure(bg=BG_BASE)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        tk.Label(win, text="Callsign:", bg=BG_BASE, fg=TEXT_MAIN,
                 font=(FONT_FAMILY, 10)).pack(pady=(16, 4))

        cs_var = tk.StringVar()
        cs_entry = tk.Entry(
            win, textvariable=cs_var, font=(FONT_FAMILY, 11),
            bg=BG_PANEL, fg=TEXT_MAIN, relief="flat", width=14,
            insertbackground=TEXT_MAIN,
        )
        cs_entry.pack()
        cs_entry.focus_set()

        cat_var = tk.StringVar(value="departure")
        cat_row = tk.Frame(win, bg=BG_BASE)
        cat_row.pack(pady=8)
        for cat, label in [("departure", "Departure"), ("arrival", "Arrival")]:
            tk.Radiobutton(
                cat_row, text=label, variable=cat_var, value=cat,
                bg=BG_BASE, fg=TEXT_MAIN, selectcolor=BG_PANEL,
                activebackground=BG_BASE, font=(FONT_FAMILY, 9),
            ).pack(side="left", padx=8)

        def _confirm():
            cs = cs_var.get().strip() or "UNKNOWN"
            self._add_strip(cat_var.get(), callsign=cs)
            win.destroy()

        win.bind("<Return>", lambda e: _confirm())
        tk.Button(
            win, text="Add Strip", bg=ACCENT, fg="white",
            font=(FONT_FAMILY, 9), relief="flat", padx=12, pady=4,
            cursor="hand2", command=_confirm,
        ).pack()

    def add_from_pilot(self, callsign: str, category: str) -> None:
        """Called from the main window when the AI identifies a flight."""
        self._add_strip(category, callsign=callsign)


# ── context window ──────────────────────────────────────────────────────────────
class ContextWindow(tk.Toplevel):
    """Window for supplying METAR and airport context to the AI model."""

    def __init__(self, parent: tk.Misc, on_save) -> None:
        super().__init__(parent)
        self.title("AI Context")
        self.geometry("520x500")
        self.minsize(400, 380)
        self.configure(bg=BG_BASE)
        self._on_save = on_save
        self._build()

    def _build(self) -> None:
        hdr = tk.Frame(self, bg=BG_BASE, padx=16, pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="AI Context", bg=BG_BASE, fg=TEXT_MAIN,
                 font=(FONT_FAMILY, 14, "bold")).pack(side="left")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG_BASE, padx=16, pady=12)
        body.pack(fill="both", expand=True)

        # METAR
        tk.Label(body, text="METAR", bg=BG_BASE, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")
        tk.Label(body, text="Paste the current METAR report for the active airport.",
                 bg=BG_BASE, fg=TEXT_DIM, font=(FONT_FAMILY, 8)).pack(anchor="w", pady=(0, 4))

        self._metar_text = tk.Text(
            body, height=5, bg=BG_CARD, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 10), relief="flat",
            insertbackground=TEXT_MAIN, wrap="word",
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self._metar_text.pack(fill="x")

        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=10)

        # Airport info
        tk.Label(body, text="Airport Information", bg=BG_BASE, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 9, "bold")).pack(anchor="w")
        tk.Label(body, text="Active runways, taxiways, procedures, NOTAMs, etc.",
                 bg=BG_BASE, fg=TEXT_DIM, font=(FONT_FAMILY, 8)).pack(anchor="w", pady=(0, 4))

        self._airport_text = tk.Text(
            body, height=7, bg=BG_CARD, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 10), relief="flat",
            insertbackground=TEXT_MAIN, wrap="word",
            highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT,
        )
        self._airport_text.pack(fill="both", expand=True)

        footer = tk.Frame(self, bg=BG_BASE, padx=16, pady=10)
        footer.pack(fill="x")

        self._status_lbl = tk.Label(footer, text="", bg=BG_BASE, fg=GREEN,
                                    font=(FONT_FAMILY, 9))
        self._status_lbl.pack(side="left")

        tk.Button(
            footer, text="Clear All", bg=BG_PANEL, fg=TEXT_DIM,
            font=(FONT_FAMILY, 9), relief="flat", padx=10, pady=4,
            cursor="hand2", command=self._clear,
        ).pack(side="right", padx=(8, 0))

        tk.Button(
            footer, text="Save Context", bg=ACCENT, fg="white",
            font=(FONT_FAMILY, 9, "bold"), relief="flat", padx=12, pady=4,
            cursor="hand2", command=self._save,
        ).pack(side="right")

    def load_context(self, metar: str, airport_info: str) -> None:
        self._metar_text.delete("1.0", "end")
        self._metar_text.insert("1.0", metar)
        self._airport_text.delete("1.0", "end")
        self._airport_text.insert("1.0", airport_info)

    def _save(self) -> None:
        metar       = self._metar_text.get("1.0", "end").strip()
        airport_inf = self._airport_text.get("1.0", "end").strip()
        self._on_save(metar, airport_inf)
        self._status_lbl.config(text="✓ Context saved", fg=GREEN)
        self.after(2000, lambda: self._status_lbl.config(text=""))

    def _clear(self) -> None:
        self._metar_text.delete("1.0", "end")
        self._airport_text.delete("1.0", "end")
        self._on_save("", "")
        self._status_lbl.config(text="Context cleared.", fg=TEXT_DIM)
        self.after(2000, lambda: self._status_lbl.config(text=""))


# ── main application ──────────────────────────────────────────────────────────
class ATCAssistant(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        # Load font first — must happen before any widget references FONT_FAMILY
        global FONT_FAMILY
        FONT_FAMILY = _load_lexend()

        self.title("PTFS ATC Assistant")
        self.geometry("680x780")
        self.minsize(520, 500)
        self.configure(bg=BG_BASE)

        self._config         = _load_config()
        _saved_model_id      = self._config.get("default_model", "llama3.2:3b")
        _saved_model_lbl     = next(
            (lbl for lbl, mid in _MODEL_OPTIONS.items() if mid == _saved_model_id),
            _DEFAULT_MODEL_LABEL,
        )

        self._audio           = AudioCapture()
        self._transcriber     = Transcriber(model_size="base.en")
        self._responder       = ATCResponder(model=_saved_model_id)

        self._devices: List[AudioDevice] = []
        self._current_device: Optional[AudioDevice] = None
        self._listening       = False
        self._using_mic       = False
        self._atc_type_var    = tk.StringVar(value="All")
        self._saved_model_lbl = _saved_model_lbl

        # Context
        self._metar_context   = self._config.get("metar_context", "")
        self._airport_context = self._config.get("airport_context", "")
        self._responder.context = self._build_context_string()

        # Secondary windows (lazily created)
        self._strip_board: Optional[FlightStripBoard] = None
        self._context_win: Optional[ContextWindow]    = None

        self._ui_queue:  queue.Queue = queue.Queue()
        self._seg_queue: queue.Queue = queue.Queue()
        self._worker_running = False

        # Pilot mode state
        self._mode_var       = tk.StringVar(value="ATC")
        self._tts_enabled    = tk.BooleanVar(value=True)
        self._pilot_history: List[dict] = []
        self._pilot_busy     = False

        self._build_ui()
        self._apply_combobox_style()
        self._load_devices_async()
        self._load_model_async()
        self._poll_ui_queue()

    def _build_context_string(self) -> str:
        parts = []
        if self._metar_context:
            parts.append(f"Current METAR: {self._metar_context}")
        if self._airport_context:
            parts.append(f"Airport Information:\n{self._airport_context}")
        return "\n\n".join(parts)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── header ──────────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_BASE, padx=16, pady=10)
        header.pack(fill="x")

        tk.Label(
            header, text="✈  ATC Assistant", bg=BG_BASE, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 13, "bold"),
        ).pack(side="left")

        # Mode toggle (segmented: ATC | Pilot)
        mode_seg = tk.Frame(header, bg=BORDER, padx=1, pady=1)
        mode_seg.pack(side="left", padx=(14, 0))

        self._atc_mode_btn = tk.Button(
            mode_seg, text="ATC", bg=ACCENT, fg="white",
            font=(FONT_FAMILY, 9, "bold"), relief="flat", padx=12, pady=3,
            cursor="hand2", command=lambda: self._switch_mode("ATC"),
        )
        self._atc_mode_btn.pack(side="left")

        self._pilot_mode_btn = tk.Button(
            mode_seg, text="Pilot", bg=BG_PANEL, fg=TEXT_DIM,
            font=(FONT_FAMILY, 9), relief="flat", padx=12, pady=3,
            cursor="hand2", command=lambda: self._switch_mode("Pilot"),
        )
        self._pilot_mode_btn.pack(side="left")

        self._status_lbl = tk.Label(
            header, text="● Loading…", bg=BG_BASE, fg=ORANGE,
            font=(FONT_FAMILY, 10),
        )
        self._status_lbl.pack(side="right")

        self._tts_btn = tk.Button(
            header, text="🔊", bg=ACCENT_LIGHT, fg=ACCENT,
            font=(FONT_FAMILY, 9), relief="flat", padx=8, pady=3,
            activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
            cursor="hand2", command=self._toggle_tts,
        )
        self._tts_btn.pack(side="right", padx=(0, 6))

        tk.Button(
            header, text="Context", bg=BG_PANEL, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 9), relief="flat", padx=8, pady=3,
            activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
            cursor="hand2", command=self._open_context,
        ).pack(side="right", padx=(0, 6))

        tk.Button(
            header, text="⊞  Strips", bg=BG_PANEL, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 9), relief="flat", padx=8, pady=3,
            activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
            cursor="hand2", command=self._open_strip_board,
        ).pack(side="right", padx=(0, 6))

        tk.Button(
            header, text="⚙  Settings", bg=BG_PANEL, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 9), relief="flat", padx=8, pady=3,
            activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
            cursor="hand2", command=self._open_settings,
        ).pack(side="right", padx=(0, 6))

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── chat area ──────────────────────────────────────────────
        self._chat = ScrollableFrame(self, bg_color=BG_CARD)
        self._chat.pack(fill="both", expand=True)

        self._add_system_msg("Select an audio source and press Start Listening.")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── controls ────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG_BASE, padx=14, pady=10)
        ctrl.pack(fill="x", side="bottom")

        # Container that swaps between ATC and Pilot controls
        mode_ctrl = tk.Frame(ctrl, bg=BG_BASE)
        mode_ctrl.pack(fill="x", pady=(0, 6))

        # ── ATC controls (device + mic + listen) ──────────────────
        self._atc_ctrl = tk.Frame(mode_ctrl, bg=BG_BASE)
        self._atc_ctrl.pack(fill="x")   # visible by default (ATC mode)

        tk.Label(self._atc_ctrl, text="Audio source:", bg=BG_BASE, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 10)).pack(side="left", padx=(0, 6))

        self._device_var = tk.StringVar(value="Scanning devices…")
        self._device_combo = ttk.Combobox(
            self._atc_ctrl, textvariable=self._device_var, width=30, state="readonly",
            font=(FONT_FAMILY, 10),
        )
        self._device_combo.pack(side="left", padx=(0, 8))
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_changed)

        self._mic_btn = tk.Button(
            self._atc_ctrl, text="🎤  Mic Off", bg=BG_PANEL, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 10), relief="flat", padx=10, pady=4,
            activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
            cursor="hand2", command=self._toggle_mic,
        )
        self._mic_btn.pack(side="left", padx=(0, 8))

        self._listen_btn = tk.Button(
            self._atc_ctrl, text="▶  Start Listening", bg=ACCENT, fg="white",
            font=(FONT_FAMILY, 10, "bold"), relief="flat", padx=12, pady=4,
            activebackground="#1558b0", activeforeground="white",
            cursor="hand2", command=self._toggle_listening,
        )
        self._listen_btn.pack(side="left")

        # ── Pilot controls (text input + quick-start buttons) ─────
        self._pilot_ctrl = tk.Frame(mode_ctrl, bg=BG_BASE)
        # NOT packed initially — shown when switching to Pilot mode

        # Input row
        pilot_input_row = tk.Frame(self._pilot_ctrl, bg=BG_BASE)
        pilot_input_row.pack(fill="x")

        self._pilot_input_var = tk.StringVar()
        self._pilot_entry = tk.Entry(
            pilot_input_row, textvariable=self._pilot_input_var,
            bg=BG_CARD, fg=TEXT_MAIN, font=(FONT_FAMILY, 11),
            relief="flat", insertbackground=TEXT_MAIN,
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        self._pilot_entry.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=5)
        self._pilot_entry.bind("<Return>", lambda _e: self._send_pilot_message())

        tk.Button(
            pilot_input_row, text="↺", bg=BG_PANEL, fg=TEXT_DIM,
            font=(FONT_FAMILY, 11), relief="flat", padx=8, pady=4,
            activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
            cursor="hand2", command=self._clear_pilot_history,
        ).pack(side="right", padx=(4, 0))

        self._send_btn = tk.Button(
            pilot_input_row, text="Send ➤", bg=ACCENT, fg="white",
            font=(FONT_FAMILY, 10, "bold"), relief="flat", padx=14, pady=5,
            activebackground="#1558b0", activeforeground="white",
            cursor="hand2", command=self._send_pilot_message,
        )
        self._send_btn.pack(side="right")

        # Quick-start buttons
        quick_row = tk.Frame(self._pilot_ctrl, bg=BG_BASE)
        quick_row.pack(fill="x", pady=(7, 0))

        _QUICK_PROMPTS = [
            ("📋 Clearance",  "Requesting IFR clearance"),
            ("🔙 Pushback",   "Requesting pushback and engine start"),
            ("🚕 Taxi",       "Ready for taxi to the runway"),
            ("🛫 Takeoff",    "Ready for departure"),
            ("🛬 Approach",   "Requesting approach clearance"),
            ("↩ Go Around",   "Going around"),
        ]

        for label, phrase in _QUICK_PROMPTS:
            tk.Button(
                quick_row, text=label, bg=BG_PANEL, fg=TEXT_MAIN,
                font=(FONT_FAMILY, 8), relief="flat", padx=8, pady=3,
                activebackground=ACCENT_LIGHT, activeforeground=ACCENT,
                cursor="hand2",
                command=lambda p=phrase: self._quick_send(p),
            ).pack(side="left", padx=(0, 4))

        # ── Shared: model + sensitivity (always visible) ──────────
        self._shared_ctrl = tk.Frame(ctrl, bg=BG_BASE)
        self._shared_ctrl.pack(fill="x", pady=(4, 0))

        tk.Label(self._shared_ctrl, text="Model:", bg=BG_BASE, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 6))

        self._model_var = tk.StringVar(value=self._saved_model_lbl)
        model_combo = ttk.Combobox(
            self._shared_ctrl, textvariable=self._model_var,
            values=list(_MODEL_OPTIONS.keys()),
            width=26, state="readonly", font=(FONT_FAMILY, 9),
        )
        model_combo.pack(side="left", padx=(0, 20))
        model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        tk.Label(self._shared_ctrl, text="Sensitivity:", bg=BG_BASE, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 8))

        self._thresh_var = tk.DoubleVar(value=0.008)
        tk.Scale(
            self._shared_ctrl, from_=0.002, to=0.05, resolution=0.001, orient="horizontal",
            variable=self._thresh_var, bg=BG_BASE, fg=TEXT_DIM,
            troughcolor=BG_PANEL, highlightthickness=0, bd=0, sliderrelief="flat",
            length=160, showvalue=False, command=self._on_threshold_changed,
        ).pack(side="left")

        self._thresh_lbl = tk.Label(
            self._shared_ctrl, text="0.008", bg=BG_BASE, fg=TEXT_DIM,
            font=(FONT_FAMILY, 9), width=5,
        )
        self._thresh_lbl.pack(side="left", padx=(4, 0))

    def _apply_combobox_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "TCombobox",
            fieldbackground=BG_CARD,
            background=BG_CARD,
            foreground=TEXT_MAIN,
            selectbackground=BG_CARD,
            selectforeground=TEXT_MAIN,
            arrowcolor=TEXT_DIM,
        )
        style.map("TCombobox", fieldbackground=[("readonly", BG_CARD)])
        style.configure(
            "TScrollbar", background=BG_PANEL, troughcolor=BG_BASE,
            arrowcolor=TEXT_DIM, darkcolor=BG_PANEL, lightcolor=BG_PANEL,
        )

    # ── window openers ─────────────────────────────────────────────────────────

    def _open_strip_board(self) -> None:
        if self._strip_board is None or not self._strip_board.winfo_exists():
            self._strip_board = FlightStripBoard(self)
        else:
            self._strip_board.lift()
            self._strip_board.focus_force()

    def _open_context(self) -> None:
        if self._context_win is None or not self._context_win.winfo_exists():
            self._context_win = ContextWindow(self, on_save=self._on_context_saved)
            self._context_win.load_context(self._metar_context, self._airport_context)
        else:
            self._context_win.lift()
            self._context_win.focus_force()

    def _on_context_saved(self, metar: str, airport_info: str) -> None:
        self._metar_context   = metar
        self._airport_context = airport_info
        self._responder.context = self._build_context_string()
        self._config["metar_context"]   = metar
        self._config["airport_context"] = airport_info
        _save_config(self._config)

    # ── async initialisation ──────────────────────────────────────────────────

    def _load_devices_async(self) -> None:
        def _run() -> None:
            devices = self._audio.list_devices()
            self._ui_queue.put(("devices", devices))

        threading.Thread(target=_run, daemon=True).start()

    def _load_model_async(self) -> None:
        def _run() -> None:
            self._ui_queue.put(("status", ("● Loading STT model…", ORANGE)))
            self._transcriber.load()

            if self._transcriber.load_error:
                self._ui_queue.put(
                    ("status", ("⚠ STT load failed", RED))
                )
                self._ui_queue.put(
                    ("sysmsg", f"faster-whisper error: {self._transcriber.load_error}")
                )
                return

            if not self._responder.is_available():
                self._ui_queue.put(("status", ("⚠ Ollama offline", RED)))
                self._ui_queue.put(
                    ("sysmsg",
                     "Ollama is not running.\n"
                     "1. Install from https://ollama.ai\n"
                     f"2. Run:  ollama pull {self._responder.model}\n"
                     "3. Then restart this app.")
                )
            elif not self._responder.model_is_pulled():
                self._ui_queue.put(("status", ("⚠ Model not found", RED)))
                self._ui_queue.put(
                    ("sysmsg",
                     f"Model {self._responder.model} is not downloaded.\n"
                     f"Run in a terminal:  ollama pull {self._responder.model}")
                )
            else:
                self._ui_queue.put(("status", ("● Ready", GREEN)))

        threading.Thread(target=_run, daemon=True).start()

    # ── worker thread (STT → LLM) ─────────────────────────────────────────────

    def _start_worker(self) -> None:
        self._worker_running = True
        threading.Thread(target=self._worker_loop, daemon=True).start()

    def _stop_worker(self) -> None:
        self._worker_running = False

    def _worker_loop(self) -> None:
        while self._worker_running:
            try:
                audio: np.ndarray = self._seg_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── transcription ──
            self._ui_queue.put(("status", ("● Transcribing…", ORANGE)))
            text = self._transcriber.transcribe(audio)

            if not text or len(text.strip()) < 3:
                self._ui_queue.put(("status", ("● Listening…", GREEN)))
                continue

            self._ui_queue.put(("pilot", text))
            self._ui_queue.put(("status", ("● Thinking…", ORANGE)))

            # Fire-and-forget AI strip parsing — runs in parallel with LLM response
            capture_text = text

            def _parse_strip(t: str = capture_text) -> None:
                strip_info = self._responder.parse_flight_info(t)
                if strip_info:
                    self._ui_queue.put(("strip", strip_info))

            threading.Thread(target=_parse_strip, daemon=True).start()

            # ── LLM response ──
            response = ""
            try:
                for token in self._responder.get_response_stream(text):
                    response += token
            except Exception as exc:
                self._ui_queue.put(("status", ("⚠ LLM error", RED)))
                self._ui_queue.put(("sysmsg", f"LLM error: {exc}"))
                continue

            if response.strip():
                self._ui_queue.put(("atc", response.strip()))

            self._ui_queue.put(("status", ("● Listening…", GREEN)))

    # ── audio callback ────────────────────────────────────────────────────────

    def _on_audio_segment(self, audio: np.ndarray) -> None:
        """Called from the audio thread when a speech segment is ready."""
        self._seg_queue.put_nowait(audio)

    # ── UI event handlers ─────────────────────────────────────────────────────

    def _open_settings(self) -> None:
        win = tk.Toplevel(self)
        win.title("Settings")
        win.configure(bg=BG_BASE)
        win.geometry("300x240")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        # Position near top-right of main window
        x = self.winfo_rootx() + self.winfo_width() - 320
        y = self.winfo_rooty() + 50
        win.geometry(f"+{x}+{y}")

        tk.Label(
            win, text="Settings", bg=BG_BASE, fg=TEXT_MAIN,
            font=(FONT_FAMILY, 12, "bold"), pady=10,
        ).pack(anchor="w", padx=18)

        tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=14, pady=(0, 10))

        tk.Label(win, text="ATC Controller Type", bg=BG_BASE, fg=TEXT_DIM,
                 font=(FONT_FAMILY, 8)).pack(anchor="w", padx=18)

        _TYPE_DESC = {
            "All":       "Handle any request",
            "Departure": "Post-takeoff: climbs, headings",
            "Ground":    "Taxi, pushback, crossings",
            "Clearance": "Pre-departure clearances",
        }

        btn_frame = tk.Frame(win, bg=BG_BASE)
        btn_frame.pack(fill="x", padx=14, pady=(6, 0))

        for atc_type, desc in _TYPE_DESC.items():
            row = tk.Frame(btn_frame, bg=BG_BASE)
            row.pack(fill="x", pady=2)
            tk.Radiobutton(
                row, text=atc_type, variable=self._atc_type_var, value=atc_type,
                bg=BG_BASE, fg=TEXT_MAIN, selectcolor=BG_PANEL,
                activebackground=BG_BASE, activeforeground=TEXT_MAIN,
                font=(FONT_FAMILY, 10), indicatoron=True,
                command=self._on_atc_type_changed,
            ).pack(side="left")
            tk.Label(row, text=desc, bg=BG_BASE, fg=TEXT_DIM,
                     font=(FONT_FAMILY, 8)).pack(side="left", padx=(6, 0))

        tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=14, pady=10)

        tk.Button(
            win, text="Close", bg=ACCENT, fg="white", font=(FONT_FAMILY, 10),
            relief="flat", padx=14, pady=4, command=win.destroy,
        ).pack()

    def _on_atc_type_changed(self) -> None:
        self._responder.atc_type = self._atc_type_var.get()
        self._add_system_msg(f"ATC type: {self._responder.atc_type}")

    # ── mode switching ────────────────────────────────────────────────────────

    def _switch_mode(self, mode: str) -> None:
        if self._mode_var.get() == mode:
            return
        self._mode_var.set(mode)
        if mode == "ATC":
            self._pilot_ctrl.pack_forget()
            self._atc_ctrl.pack(fill="x")
            self._atc_mode_btn.config(bg=ACCENT, fg="white",
                                      font=(FONT_FAMILY, 9, "bold"))
            self._pilot_mode_btn.config(bg=BG_PANEL, fg=TEXT_DIM,
                                        font=(FONT_FAMILY, 9))
            self._responder.atc_type = self._atc_type_var.get()
            self._add_system_msg("ATC mode — listening to audio.")
        else:
            # Stop listening when entering Pilot mode
            if self._listening:
                self._stop_listening()
            self._atc_ctrl.pack_forget()
            self._pilot_ctrl.pack(fill="x")
            self._atc_mode_btn.config(bg=BG_PANEL, fg=TEXT_DIM,
                                      font=(FONT_FAMILY, 9))
            self._pilot_mode_btn.config(bg=ACCENT, fg="white",
                                        font=(FONT_FAMILY, 9, "bold"))
            self._responder.atc_type = "Pilot"
            self._add_system_msg(
                "Pilot mode — type your radio calls below, "
                "or use the quick-start buttons.\n"
                "Use ↺ to start a new practice session."
            )
            self._pilot_entry.focus_set()

    def _toggle_tts(self) -> None:
        enabled = not self._tts_enabled.get()
        self._tts_enabled.set(enabled)
        if enabled:
            self._tts_btn.config(bg=ACCENT_LIGHT, fg=ACCENT, text="🔊")
        else:
            tts.stop()
            self._tts_btn.config(bg=BG_PANEL, fg=TEXT_DIM, text="🔇")

    def _send_pilot_message(self) -> None:
        text = self._pilot_input_var.get().strip()
        if not text or self._pilot_busy:
            return
        self._pilot_input_var.set("")
        self._pilot_busy = True
        self._send_btn.config(state="disabled")
        self._pilot_history.append({"role": "user", "content": text})
        self._add_bubble(text, is_pilot=True)
        self._update_status("● Thinking…", ORANGE)

        capture_history = list(self._pilot_history)

        def _run() -> None:
            response = ""
            try:
                for token in self._responder.get_response_with_history(capture_history):
                    response += token
            except Exception as exc:
                self._ui_queue.put(("status", ("⚠ LLM error", RED)))
                self._ui_queue.put(("sysmsg", f"LLM error: {exc}"))
                self._ui_queue.put(("pilot_done", None))
                return

            if response.strip():
                resp = response.strip()
                self._pilot_history.append({"role": "assistant", "content": resp})
                self._ui_queue.put(("atc", resp))

            self._ui_queue.put(("status", ("● Ready", GREEN)))
            self._ui_queue.put(("pilot_done", None))

        threading.Thread(target=_run, daemon=True).start()

    def _quick_send(self, phrase: str) -> None:
        self._pilot_input_var.set(phrase)
        self._send_pilot_message()

    def _clear_pilot_history(self) -> None:
        self._pilot_history.clear()
        self._add_system_msg("Practice session reset.")

    def _on_model_changed(self, _event: Optional[tk.Event]) -> None:
        selected = self._model_var.get()
        model_id = _MODEL_OPTIONS.get(selected, selected.split("  ")[0])
        self._responder.model = model_id
        self._config["default_model"] = model_id
        _save_config(self._config)
        self._add_system_msg(f"Model switched to {model_id}")

    def _on_threshold_changed(self, _val: str) -> None:
        v = round(self._thresh_var.get(), 3)
        self._thresh_lbl.config(text=f"{v:.3f}")
        self._audio.threshold = v

    def _on_device_changed(self, _event: Optional[tk.Event]) -> None:
        selected = self._device_var.get()
        for d in self._devices:
            if str(d) == selected:
                self._current_device = d
                self._using_mic = not d.is_loopback
                self._mic_btn.config(
                    text="🎤  Mic ON" if self._using_mic else "🎤  Mic Off",
                    bg=ACCENT_LIGHT if self._using_mic else BG_PANEL,
                    fg=ACCENT if self._using_mic else TEXT_MAIN,
                )
                self._config["default_device"] = selected
                _save_config(self._config)
                if self._listening:
                    self._audio.start(d, self._on_audio_segment)
                break

    def _toggle_mic(self) -> None:
        """Switch between microphone and loopback."""
        if self._using_mic:
            # Switch back to first loopback device
            for d in self._devices:
                if d.is_loopback:
                    self._device_var.set(str(d))
                    self._on_device_changed(None)
                    return
            # No loopback available — stay on mic
        else:
            # Switch to first microphone
            for d in self._devices:
                if not d.is_loopback:
                    self._device_var.set(str(d))
                    self._on_device_changed(None)
                    return

    def _toggle_listening(self) -> None:
        if self._listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self) -> None:
        if not self._current_device:
            self._add_system_msg("No audio device selected.")
            return
        if not self._transcriber.loaded:
            self._add_system_msg("STT model is still loading — please wait.")
            return

        self._listening = True
        self._listen_btn.config(text="⏹  Stop Listening", bg=RED,
                                activebackground="#b71c1c")
        self._audio.start(self._current_device, self._on_audio_segment)
        self._start_worker()
        self._update_status("● Listening…", GREEN)

    def _stop_listening(self) -> None:
        self._listening = False
        self._listen_btn.config(text="▶  Start Listening", bg=ACCENT,
                                activebackground="#1558b0")
        self._audio.stop()
        self._stop_worker()
        self._update_status("● Ready", GREEN)

    # ── UI queue polling ──────────────────────────────────────────────────────

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, data = self._ui_queue.get_nowait()

                if kind == "devices":
                    devices: List[AudioDevice] = data
                    self._devices = devices
                    names = [str(d) for d in devices]
                    self._device_combo["values"] = names
                    if names:
                        saved_device = self._config.get("default_device", "")
                        default = (
                            saved_device if saved_device in names
                            else next((str(d) for d in devices if d.is_loopback), names[0])
                        )
                        self._device_var.set(default)
                        self._current_device = next(
                            (d for d in devices if str(d) == default), devices[0]
                        )
                        self._using_mic = not self._current_device.is_loopback
                    if not HAS_WASAPI:
                        self._add_system_msg(
                            "pyaudiowpatch not found — loopback capture unavailable.\n"
                            "Install with:  pip install pyaudiowpatch"
                        )

                elif kind == "status":
                    text, color = data
                    self._update_status(text, color)

                elif kind == "pilot":
                    self._add_bubble(data, is_pilot=True)

                elif kind == "atc":
                    self._add_bubble(data, is_pilot=False)
                    if self._tts_enabled.get():
                        tts.speak(data)

                elif kind == "pilot_done":
                    self._pilot_busy = False
                    self._send_btn.config(state="normal")

                elif kind == "sysmsg":
                    self._add_system_msg(data)

                elif kind == "strip":
                    callsign, category = data
                    if self._strip_board and self._strip_board.winfo_exists():
                        self._strip_board.add_from_pilot(callsign, category)

        except queue.Empty:
            pass

        self.after(40, self._poll_ui_queue)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _update_status(self, text: str, color: str) -> None:
        self._status_lbl.config(text=text, fg=color)

    def _add_bubble(self, text: str, is_pilot: bool) -> None:
        bubble = ChatBubble(self._chat.inner, text, is_pilot=is_pilot)
        bubble.pack(fill="x", pady=2)
        self._chat.scroll_to_bottom()

    def _add_system_msg(self, text: str) -> None:
        frame = tk.Frame(self._chat.inner, bg=BG_CARD)
        frame.pack(fill="x", padx=20, pady=6)
        tk.Label(
            frame, text=text, bg=BG_CARD, fg=TEXT_DIM,
            font=(FONT_FAMILY, 9, "italic"), wraplength=540, justify="center",
        ).pack()
        self._chat.scroll_to_bottom()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def on_closing(self) -> None:
        self._audio.stop()
        self._worker_running = False
        self.destroy()


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ATCAssistant()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
