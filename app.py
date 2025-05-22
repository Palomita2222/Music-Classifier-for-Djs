import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
from files.genre_classifier import GenreClassifier
from files.spectogram_generator import SpectrogramGenerator
import os
import shutil
import librosa
import threading
import io
import csv
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC
import pickle
import time

class GenreApp:
    def __init__(self, model_path, class_names):
        self.classifier = GenreClassifier(model_path, class_names)
        self.spectrogram = SpectrogramGenerator()
        self.tracks = []
        self.track_checkboxes = {}
        self.csv_path = "files\data.csv"

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("🎵 Music Analyzer & Classifier V1.3")
        self.root.geometry("600x900")

        self.title_label = ctk.CTkLabel( #Title
            self.root,
            text="🎶 Music Analyzer & Classifier",
            font=("Arial", 26, "bold"),
            text_color="#1f6aa5"
        )
        self.title_label.pack(pady=(20, 5))

        self.label = ctk.CTkLabel(
            self.root,
            text="Select MP3 files",
            font=("Arial", 18)
        )
        self.label.pack(pady=5) #Title

        self.select_button = ctk.CTkButton(self.root, text="Browse", command=self.load_files)
        self.select_button.pack(pady=5)

        self.filters_frame = ctk.CTkFrame(self.root)
        self.filters_frame.pack(pady=10)

        self.genre_var = ctk.StringVar(value="All")
        self.genre_menu = ctk.CTkOptionMenu(self.filters_frame, variable=self.genre_var, values=["All"])
        self.genre_menu.pack(side="left", padx=5)

        self.harmonic_var = ctk.BooleanVar(value=False)
        self.harmonic_check = ctk.CTkCheckBox(self.filters_frame, text="Harmonic Mix", variable=self.harmonic_var)
        self.harmonic_check.pack(side="left", padx=5)

        self.sort_bpm_var = ctk.BooleanVar(value=False)
        self.sort_bpm_check = ctk.CTkCheckBox(self.filters_frame, text="Sort BPM ↑", variable=self.sort_bpm_var)
        self.sort_bpm_check.pack(side="left", padx=5)

        self.apply_btn = ctk.CTkButton(self.filters_frame, text="Apply Filters", command=self.apply_filters)
        self.apply_btn.pack(side="left", padx=5)

        self.bottom_frame = ctk.CTkFrame(self.root)
        self.bottom_frame.pack(pady=10)

        self.export_btn = ctk.CTkButton(self.bottom_frame, text="📤 Export Playlist", command=self.export_playlist)
        self.export_btn.pack(side="left", padx=10)

        self.clear_btn = ctk.CTkButton(self.bottom_frame, text="🗑️ Clear Cache", command=self.clear_cache)
        self.clear_btn.pack(side="left", padx=10)

        self.results_box = ctk.CTkScrollableFrame(self.root, width=580, height=600)
        self.results_box.pack(pady=20)

        self.load_cached_tracks()

    def load_cached_tracks(self):
        if os.path.exists(self.csv_path):
            with open(self.csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.tracks.append({
                        "filename": row["filename"],
                        "filepath": row["filepath"],
                        "genre": row["genre"],
                        "confidence": float(row["confidence"]),
                        "bpm": float(row["bpm"].replace('[', '').replace(']', '')),
                        "key": row["key"]
                    })

        if self.tracks:
            genres = list(set([t['genre'] for t in self.tracks]))
            self.genre_menu.configure(values=["All"] + genres)
            self.apply_filters()

    def save_to_csv(self):
        with open(self.csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "filepath", "genre", "confidence", "bpm", "key"])
            writer.writeheader()
            for t in self.tracks:
                writer.writerow(t)

    def load_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("MP3 files", "*.mp3")])
        if file_paths:
            threading.Thread(target=self.process_files, args=(file_paths,)).start()

    def process_files(self, file_paths):
        existing_paths = {t["filepath"] for t in self.tracks}
        new_files = [f for f in file_paths if f not in existing_paths]

        if not new_files:
            return

        for file_path in new_files:
            filename = os.path.basename(file_path)
            try:
                spec_path = r"files\temp_spec.png"
                self.spectrogram.generate(file_path, spec_path)

                genre, confidence = self.classifier.predict(spec_path)

                y, sr = librosa.load(file_path, sr=None, mono=True)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                key_index = chroma.mean(axis=1).argmax()
                key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                key = key_names[key_index]

                self.tracks.append({
                    "filename": filename,
                    "filepath": file_path,
                    "genre": genre,
                    "confidence": confidence,
                    "bpm": float(tempo),
                    "key": key
                })

            except Exception as e:
                error_label = ctk.CTkLabel(self.results_box, text=f"⚠️ {filename}\nError: {e}", anchor="w", justify="left")
                error_label.pack(pady=10, anchor="w")

        genres = list(set([track['genre'] for track in self.tracks]))
        self.genre_menu.configure(values=["All"] + genres)
        self.save_to_csv()
        self.apply_filters()

    def apply_filters(self):
        for widget in self.results_box.winfo_children():
            widget.destroy()
        self.track_checkboxes.clear()

        selected_genre = self.genre_var.get()
        harmonic_only = self.harmonic_var.get()
        sort_bpm = self.sort_bpm_var.get()

        filtered = self.tracks

        if selected_genre != "All":
            filtered = [t for t in filtered if t["genre"] == selected_genre]

        if harmonic_only and filtered:
            used = []
            pool = filtered.copy()
            current = pool.pop(0)
            used.append(current)

            while pool:
                current_key = self.key_to_camelot(current['key'])
                compatibles = self.get_compatible_keys(current['key'])

                match = None
                for t in pool:
                    if self.key_to_camelot(t['key']) in compatibles:
                        match = t
                        break

                if not match:
                    def camelot_distance(a, b):
                        def key_val(k): return int(k[:-1])
                        a_val = key_val(a)
                        b_val = key_val(b)
                        return min(abs(a_val - b_val), 12 - abs(a_val - b_val))

                    current_val = self.key_to_camelot(current['key'])
                    pool.sort(key=lambda x: camelot_distance(current_val, self.key_to_camelot(x['key'])))
                    match = pool[0]

                pool.remove(match)
                used.append(match)
                current = match

            filtered = used

        if sort_bpm:
            filtered = sorted(filtered, key=lambda x: x["bpm"])

        for track in filtered:
            self.render_track(track)

    def render_track(self, track):
        selected_color = "#cceeff"
        hover_color = "#e8f6ff"
        default_color = "#f0f0f0"

        frame = ctk.CTkFrame(self.results_box, fg_color=default_color)
        frame.pack(pady=10, fill="x", padx=10)

        self.track_checkboxes[track["filepath"]] = False

        def toggle_selection(event=None):
            selected = self.track_checkboxes[track["filepath"]]
            self.track_checkboxes[track["filepath"]] = not selected
            frame.configure(fg_color=selected_color if not selected else default_color)

        def on_enter(event):
            if not self.track_checkboxes[track["filepath"]]:
                frame.configure(fg_color=hover_color)

        def on_leave(event):
            if not self.track_checkboxes[track["filepath"]]:
                frame.configure(fg_color=default_color)

        for widget in [frame]:
            widget.bind("<Button-1>", toggle_selection)
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)

        cover_img = self.extract_cover(track["filepath"])
        if cover_img:
            cover_img = cover_img.resize((100, 100))
            cover_photo = ImageTk.PhotoImage(cover_img)
            image_label = ctk.CTkLabel(frame, image=cover_photo, text="")
            image_label.image = cover_photo
            image_label.pack(side="left", padx=10)
            image_label.bind("<Button-1>", toggle_selection)
            image_label.bind("<Enter>", on_enter)
            image_label.bind("<Leave>", on_leave)

        text = (
            f"🎵 {track['filename']}\n"
            f"• Genre: {track['genre']} ({track['confidence'] * 100:.2f}%)\n"
            f"• BPM: {int(track['bpm'])}\n"
            f"• Key: {track['key']}"
        )
        text_label = ctk.CTkLabel(frame, text=text, anchor="w", justify="left")
        text_label.pack(side="left", padx=10)
        text_label.bind("<Button-1>", toggle_selection)
        text_label.bind("<Enter>", on_enter)
        text_label.bind("<Leave>", on_leave)

    def export_playlist(self):
        selected_files = [
            path for path, selected in self.track_checkboxes.items() if selected
        ]
        if not selected_files:
            return

        base_folder = filedialog.askdirectory(title="Choose Export Destination")
        if not base_folder:
            return

        export_folder = os.path.join(base_folder, f"Playlist_Export_{int(time.time())}")
        os.makedirs(export_folder, exist_ok=True)

        for path in selected_files:
            try:
                shutil.copy2(path, export_folder)
            except Exception as e:
                print(f"Error exporting {path}: {e}")

    def clear_cache(self):
        self.tracks.clear()
        self.track_checkboxes.clear()
        for widget in self.results_box.winfo_children():
            widget.destroy()
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def extract_cover(self, mp3_path):
        try:
            audio = MP3(mp3_path, ID3=ID3)
            for tag in audio.tags.values():
                if isinstance(tag, APIC):
                    return Image.open(io.BytesIO(tag.data))
        except Exception:
            return None
        return None

    def key_to_camelot(self, key):
        key_map = {
            'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B', 'E': '12B',
            'F': '7B', 'F#': '2B', 'G': '9B', 'G#': '4B', 'A': '11B',
            'A#': '6B', 'B': '1B'
        }
        return key_map.get(key, "Unknown")

    def get_compatible_keys(self, base_key):
        camelot = self.key_to_camelot(base_key)
        if camelot == "Unknown":
            return []
        num = int(camelot[:-1])
        mode = camelot[-1]
        return [
            f"{num}{mode}",
            f"{(num - 1) % 12 or 12}{mode}",
            f"{(num + 1) % 12 or 12}{mode}",
            f"{num}{'A' if mode == 'B' else 'B'}"
        ]

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    model_path = "files\genre_classifier_mobilenetv2.h5"
    with open("files\label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)
    app = GenreApp(model_path, class_names)
    app.run()
