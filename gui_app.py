import tkinter as tk
from tkinter import messagebox
import subprocess
import threading
import queue
import time
import sys
import os


class SignDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bangla Sign Language Detector")
        self.root.geometry("600x400")
        self.root.resizable(False, False)

        # Styling
        self.root.configure(bg="#2c3e50")
        font_large = ("Helvetica", 24, "bold")
        font_medium = ("Helvetica", 16)
        button_font = ("Helvetica", 14, "bold")

        self.title_label = tk.Label(root, text="Bangla Sign Language Detector", font=("Helvetica", 28, "bold"),
                                    fg="#ecf0f1", bg="#2c3e50")
        self.title_label.pack(pady=20)

        self.sign_display_label = tk.Label(root, text="No Sign Detected", font=font_large, fg="#f39c12", bg="#34495e",
                                           wraplength=500)
        self.sign_display_label.pack(pady=30, padx=20, fill=tk.X)

        self.status_label = tk.Label(root, text="Ready", font=font_medium, fg="#bdc3c7", bg="#2c3e50")
        self.status_label.pack(pady=10)

        button_frame = tk.Frame(root, bg="#2c3e50")
        button_frame.pack(pady=20)

        self.start_button = tk.Button(button_frame, text="Start Detection", command=self.start_detection,
                                      font=button_font, bg="#27ae60", fg="white", activebackground="#2ecc71",
                                      padx=15, pady=8, relief=tk.RAISED, bd=3, cursor="hand2")
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(button_frame, text="Stop Detection", command=self.stop_detection,
                                     font=button_font, bg="#c0392b", fg="white", activebackground="#e74c3c",
                                     padx=15, pady=8, relief=tk.RAISED, bd=3, cursor="hand2", state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=10)

        self.detection_process = None
        self.queue = queue.Queue()
        self.running = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_detection(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Starting webcam...")
            self.sign_display_label.config(text="Initializing...")

            script_path = os.path.join(os.path.dirname(__file__), 'predict_sign.py')

            if not os.path.exists(script_path):
                messagebox.showerror("Error", f"predict_sign.py not found at {script_path}")
                self.stop_detection()
                return

            try:
                self.detection_process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                self.status_label.config(text="Webcam started. Detecting...")
                self.read_thread = threading.Thread(target=self._read_output, daemon=True)
                self.read_thread.start()
                self.root.after(100, self._process_queue)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start detection: {e}")
                self.stop_detection()

    def stop_detection(self):
        if self.running:
            self.running = False
            self.status_label.config(text="Stopping detection...")
            self.sign_display_label.config(text="Stopping...")

            if self.detection_process:
                try:

                    self.detection_process.terminate()
                    self.detection_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.detection_process.kill()
                except Exception as e:
                    print(f"Error terminating process: {e}")
                self.detection_process = None

            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Ready")
            self.sign_display_label.config(text="Detection Stopped")

    def _read_output(self):

        while self.running and self.detection_process and self.detection_process.stdout:
            line = self.detection_process.stdout.readline()
            if line:
                self.queue.put(line.strip())
            else:
                if not self.running:
                    break
                self.queue.put("ERROR: Detection process ended unexpectedly.")
                break

        if self.detection_process and self.detection_process.stderr:
            stderr_output = self.detection_process.stderr.read()
            if stderr_output:
                self.queue.put(f"ERROR_STDERR:{stderr_output.strip()}")

    def _process_queue(self):

        try:
            while True:
                line = self.queue.get_nowait()
                if line.startswith("WEBCAM_STARTED"):
                    self.status_label.config(text="Webcam active. Detecting signs...")
                elif line.startswith("DETECTED_SIGNS:"):
                    signs_str = line.split(":", 1)[1]
                    if signs_str == "None":
                        self.sign_display_label.config(text="No Sign Detected")
                    else:

                        display_text = signs_str.replace(";", ", ")
                        self.sign_display_label.config(text=display_text)
                elif line.startswith("ERROR:"):
                    self.status_label.config(text=line)
                    messagebox.showerror("Detection Error", line)
                    self.stop_detection()
                elif line.startswith("ERROR_STDERR:"):
                    self.status_label.config(text="Error from subprocess.")
                    messagebox.showerror("Subprocess Error", line.replace("ERROR_STDERR:", ""))
                    self.stop_detection()
                self.queue.task_done()
        except queue.Empty:
            pass

        if self.running:
            self.root.after(100, self._process_queue)
        elif self.detection_process and self.detection_process.poll() is not None:
            self.stop_detection()
            self.status_label.config(text="Ready (Process Exited)")

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            self.stop_detection()
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = SignDetectorApp(root)
    root.mainloop()
