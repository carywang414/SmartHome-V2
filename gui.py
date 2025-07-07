import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import sys
import multiprocessing
import main

def start_recognition_process():
    p = multiprocessing.Process(target=main.main)
    p.start()

def open_log():
    log_path = "log.xlsx"
    if os.path.exists(log_path):
        try:
            if os.name == 'nt':
                os.startfile(log_path)
            elif os.name == 'posix':
                subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', log_path])
        except Exception as e:
            messagebox.showerror("Error", f"無法打開 log.xlsx\n{e}")
    else:
        messagebox.showinfo("Log Not Found", "尚未產生 log.xlsx 記錄檔")

def create_gui():
    root = tk.Tk()
    root.title("Face Recognition System")
    root.geometry("300x200")
    root.resizable(False, False)

    label = tk.Label(root, text="Face Recognition", font=("Arial", 16))
    label.pack(pady=20)

    start_btn = tk.Button(root, text="▶ Start Recognition", width=25, height=2,
                          command=start_recognition_process)
    start_btn.pack(pady=5)

    log_btn = tk.Button(root, text=" View Log", width=25, height=2,
                        command=open_log)
    log_btn.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # for macOS
    create_gui()