"""
main.py - PTFS ATC Assistant
Chat-style UI that listens to system audio or microphone, transcribes pilot
transmissions with faster-whisper, and suggests ATC responses via llama3.2:3b.

Run:  python main.py
"""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import numpy as np

from audio import AudioCapture, AudioDevice, HAS_WASAPI
from llm import ATCResponder
from stt import Transcriber

# ── colour palette ────────────────────────────────────────────────────────────
BG_DARK   = "#12121f"
BG_MID    = "#1a1a2e"
BG_LIGHT  = "#22223b"
PILOT_BG  = "#2d2d44"
ATC_BG    = "#1a4fa3"
TEXT_DIM  = "#7070a0"
TEXT_MAIN = "#dde0ff"
ACCENT    = "#4e6ef2"
GREEN     = "#4caf50"
ORANGE    = "#f0a500"
RED       = "#e53935"

FONT_UI   = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_CHAT = ("Segoe UI", 11)
FONT_SMALL= ("Segoe UI", 8)


# ── scrollable frame ─────────────────────────────────────────────────────────
class ScrollableFrame(tk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        bg = kwargs.get("bg", BG_MID)

        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        self.inner = tk.Frame(self._canvas, bg=bg)
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


# ── chat bubble ───────────────────────────────────────────────────────────────
class ChatBubble(tk.Frame):
    """A single chat message bubble.  is_pilot=True → left (gray), False → right (blue)."""

    def __init__(
        self,
        parent: tk.Widget,
        text: str,
        is_pilot: bool = True,
    ) -> None:
        super().__init__(parent, bg=BG_MID)
        self._text = text

        bubble_bg  = PILOT_BG if is_pilot else ATC_BG
        label_text = "Pilot" if is_pilot else "ATC Suggestion"
        anchor     = "w"     if is_pilot else "e"

        row = tk.Frame(self, bg=BG_MID)
        row.pack(fill="x", padx=12, pady=(4, 0))

        # Small label above bubble
        tk.Label(
            row, text=label_text, bg=BG_MID, fg=TEXT_DIM, font=FONT_SMALL
        ).pack(anchor=anchor)

        # The bubble itself
        bubble = tk.Frame(row, bg=bubble_bg, padx=14, pady=9)
        bubble.pack(anchor=anchor)

        tk.Label(
            bubble,
            text=text,
            bg=bubble_bg,
            fg=TEXT_MAIN,
            font=FONT_CHAT,
            wraplength=420,
            justify="left",
        ).pack(anchor="w")

        # Copy button — ATC suggestions only
        if not is_pilot:
            copy_lbl = tk.Label(
                row, text="⧈ copy", bg=BG_MID, fg=TEXT_DIM,
                font=("Segoe UI", 8), cursor="hand2"
            )
            copy_lbl.pack(anchor="e", pady=(2, 0))
            copy_lbl.bind("<Button-1>", self._copy)
            copy_lbl.bind("<Enter>", lambda e: copy_lbl.config(fg=TEXT_MAIN))
            copy_lbl.bind("<Leave>", lambda e: copy_lbl.config(fg=TEXT_DIM))

    def _copy(self, _event: tk.Event) -> None:
        root = self.winfo_toplevel()
        root.clipboard_clear()
        root.clipboard_append(self._text)


# ── main application ──────────────────────────────────────────────────────────
class ATCAssistant(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("PTFS ATC Assistant")
        self.geometry("660x760")
        self.minsize(520, 500)
        self.configure(bg=BG_DARK)

        # Backend
        self._audio      = AudioCapture()
        self._transcriber = Transcriber(model_size="base.en")
        self._responder  = ATCResponder(model="llama3.2:3b")

        self._devices: List[AudioDevice] = []
        self._current_device: Optional[AudioDevice] = None
        self._listening = False
        self._using_mic = False
        self._atc_type_var = tk.StringVar(value="All")

        # Thread communication
        self._ui_queue: queue.Queue = queue.Queue()
        self._seg_queue: queue.Queue = queue.Queue()
        self._worker_running = False

        self._build_ui()
        self._apply_combobox_style()
        self._load_devices_async()
        self._load_model_async()
        self._poll_ui_queue()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── header ─────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_DARK, padx=18, pady=10)
        header.pack(fill="x")

        tk.Label(
            header, text="✈  ATC Assistant", bg=BG_DARK, fg=TEXT_MAIN, font=("Segoe UI", 13, "bold")
        ).pack(side="left")

        self._status_lbl = tk.Label(
            header, text="● Loading…", bg=BG_DARK, fg=ORANGE, font=FONT_UI
        )
        self._status_lbl.pack(side="right")

        tk.Button(
            header, text="⚙️  Settings", bg=BG_LIGHT, fg=TEXT_MAIN,
            font=FONT_SMALL, relief="flat", padx=8, pady=3,
            activebackground=ACCENT, activeforeground="white",
            command=self._open_settings,
        ).pack(side="right", padx=(0, 10))

        # ── chat area ──────────────────────────────────────────
        self._chat = ScrollableFrame(self, bg=BG_MID)
        self._chat.pack(fill="both", expand=True, padx=0, pady=(0, 0))

        self._add_system_msg("Select an audio source and press Start Listening.")

        # ── controls ───────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG_DARK, padx=14, pady=10)
        ctrl.pack(fill="x", side="bottom")

        # Row 0: device selector + buttons
        row0 = tk.Frame(ctrl, bg=BG_DARK)
        row0.pack(fill="x", pady=(0, 6))

        tk.Label(row0, text="Audio source:", bg=BG_DARK, fg=TEXT_DIM, font=FONT_UI).pack(
            side="left", padx=(0, 6)
        )

        self._device_var = tk.StringVar(value="Scanning devices…")
        self._device_combo = ttk.Combobox(
            row0, textvariable=self._device_var, width=32, state="readonly", font=FONT_UI
        )
        self._device_combo.pack(side="left", padx=(0, 8))
        self._device_combo.bind("<<ComboboxSelected>>", self._on_device_changed)

        self._mic_btn = tk.Button(
            row0, text="🎤  Mic Off", bg=BG_LIGHT, fg=TEXT_MAIN,
            font=FONT_UI, relief="flat", padx=10, pady=4,
            activebackground=ACCENT, activeforeground="white",
            command=self._toggle_mic,
        )
        self._mic_btn.pack(side="left", padx=(0, 8))

        self._listen_btn = tk.Button(
            row0, text="▶  Start Listening", bg=ACCENT, fg="white",
            font=FONT_BOLD, relief="flat", padx=12, pady=4,
            activebackground="#6080ff", activeforeground="white",
            command=self._toggle_listening,
        )
        self._listen_btn.pack(side="left")

        # Row 1: model selector + sensitivity slider
        row1 = tk.Frame(ctrl, bg=BG_DARK)
        row1.pack(fill="x", pady=(4, 0))

        tk.Label(row1, text="Model:", bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(
            side="left", padx=(0, 6)
        )

        self._model_var = tk.StringVar(value="llama3.2:3b  (Fast)")
        model_combo = ttk.Combobox(
            row1,
            textvariable=self._model_var,
            values=[
                "llama3.2:3b  (Fast)",
                "llama3.1:8b  (Balanced)",
                "llama3.2-vision:11b  (Vision)",
            ],
            width=26,
            state="readonly",
            font=FONT_SMALL,
        )
        model_combo.pack(side="left", padx=(0, 20))
        model_combo.bind("<<ComboboxSelected>>", self._on_model_changed)

        tk.Label(row1, text="Sensitivity:", bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL).pack(
            side="left", padx=(0, 8)
        )

        self._thresh_var = tk.DoubleVar(value=0.008)
        scale = tk.Scale(
            row1, from_=0.002, to=0.05, resolution=0.001, orient="horizontal",
            variable=self._thresh_var, bg=BG_DARK, fg=TEXT_DIM,
            troughcolor=BG_LIGHT, highlightthickness=0, bd=0, sliderrelief="flat",
            length=180, showvalue=False, command=self._on_threshold_changed,
        )
        scale.pack(side="left")

        self._thresh_lbl = tk.Label(row1, text="0.008", bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL, width=5)
        self._thresh_lbl.pack(side="left", padx=(4, 0))

    def _apply_combobox_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "TCombobox",
            fieldbackground=BG_LIGHT,
            background=BG_LIGHT,
            foreground=TEXT_MAIN,
            selectbackground=BG_LIGHT,
            selectforeground=TEXT_MAIN,
            arrowcolor=TEXT_DIM,
        )
        style.map("TCombobox", fieldbackground=[("readonly", BG_LIGHT)])
        style.configure(
            "TScrollbar", background=BG_LIGHT, troughcolor=BG_MID,
            arrowcolor=TEXT_DIM, darkcolor=BG_LIGHT, lightcolor=BG_LIGHT,
        )

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
                     "2. Run:  ollama pull llama3.2:3b\n"
                     "3. Then restart this app.")
                )
            elif not self._responder.model_is_pulled():
                self._ui_queue.put(("status", ("⚠ Model not found", RED)))
                self._ui_queue.put(
                    ("sysmsg",
                     "Model llama3.2:3b is not downloaded.\n"
                     "Run in a terminal:  ollama pull llama3.2:3b")
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
        win.configure(bg=BG_DARK)
        win.geometry("300x240")
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()

        # Position near top-right of main window
        x = self.winfo_rootx() + self.winfo_width() - 320
        y = self.winfo_rooty() + 50
        win.geometry(f"+{x}+{y}")

        tk.Label(
            win, text="Settings", bg=BG_DARK, fg=TEXT_MAIN,
            font=("Segoe UI", 12, "bold"), pady=10,
        ).pack(anchor="w", padx=18)

        # Divider
        tk.Frame(win, bg=BG_LIGHT, height=1).pack(fill="x", padx=14, pady=(0, 10))

        # ATC type
        tk.Label(
            win, text="ATC Controller Type", bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL,
        ).pack(anchor="w", padx=18)

        _TYPE_DESC = {
            "All":       "Handle any request",
            "Departure": "Post-takeoff: climbs, headings",
            "Ground":    "Taxi, pushback, crossings",
            "Clearance": "Pre-departure clearances",
        }

        btn_frame = tk.Frame(win, bg=BG_DARK)
        btn_frame.pack(fill="x", padx=14, pady=(6, 0))

        for atc_type, desc in _TYPE_DESC.items():
            row = tk.Frame(btn_frame, bg=BG_DARK)
            row.pack(fill="x", pady=2)
            tk.Radiobutton(
                row, text=atc_type, variable=self._atc_type_var, value=atc_type,
                bg=BG_DARK, fg=TEXT_MAIN, selectcolor=BG_LIGHT,
                activebackground=BG_DARK, activeforeground=TEXT_MAIN,
                font=FONT_UI, indicatoron=True,
                command=self._on_atc_type_changed,
            ).pack(side="left")
            tk.Label(
                row, text=desc, bg=BG_DARK, fg=TEXT_DIM, font=FONT_SMALL,
            ).pack(side="left", padx=(6, 0))

        tk.Frame(win, bg=BG_LIGHT, height=1).pack(fill="x", padx=14, pady=10)

        tk.Button(
            win, text="Close", bg=ACCENT, fg="white", font=FONT_UI,
            relief="flat", padx=14, pady=4,
            command=win.destroy,
        ).pack()

    def _on_atc_type_changed(self) -> None:
        self._responder.atc_type = self._atc_type_var.get()
        self._add_system_msg(f"ATC type: {self._responder.atc_type}")

    def _on_model_changed(self, _event: Optional[tk.Event]) -> None:
        selected = self._model_var.get()
        model_id = selected.split("  ")[0]  # strip the display label
        self._responder.model = model_id
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
                    bg=ACCENT if self._using_mic else BG_LIGHT,
                )
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
        self._listen_btn.config(text="⏹  Stop Listening", bg=RED, activebackground="#ff5555")
        self._audio.start(self._current_device, self._on_audio_segment)
        self._start_worker()
        self._update_status("● Listening…", GREEN)

    def _stop_listening(self) -> None:
        self._listening = False
        self._listen_btn.config(text="▶  Start Listening", bg=ACCENT, activebackground="#6080ff")
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
                        # Prefer first loopback; fall back to first mic
                        default = next((str(d) for d in devices if d.is_loopback), names[0])
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

                elif kind == "sysmsg":
                    self._add_system_msg(data)

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
        frame = tk.Frame(self._chat.inner, bg=BG_MID)
        frame.pack(fill="x", padx=20, pady=6)
        tk.Label(
            frame, text=text, bg=BG_MID, fg=TEXT_DIM,
            font=("Segoe UI", 9, "italic"), wraplength=540, justify="center",
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
