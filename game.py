import tkinter as tk
import threading
import subprocess
import queue
import sys
import os
import random
import time

# Define note heights
LOW = 50
MID = 150
HIGH = 250

# Define the song with notes and their time offsets
song = [
    {"height": None, "time_offset": 0},  # No note at start
    {"height": LOW, "time_offset": 6},   # Note at 6s
    {"height": MID, "time_offset": 8},   # Note at 8s
    {"height": HIGH, "time_offset": 10}, # Note at 10s
    {"height": LOW, "time_offset": 12},  # Note at 12s
    {"height": MID, "time_offset": 14},  # Note at 14s
    {"height": HIGH, "time_offset": 16}, # Note at 16s
    {"height": LOW, "time_offset": 18},  # Note at 18s
    {"height": HIGH, "time_offset": 20}, # Note at 20s
]

current_time = 0  # Start time for the game
current_note = None  # Track the current note being displayed

class DinoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Rhythm Run")

        # Set the game window size
        self.canvas = tk.Canvas(self.root, width=1200, height=400, bg="white")
        self.canvas.pack()

        # Create the Dino (Green Square)
        self.dino = self.canvas.create_rectangle(75, 150, 125, 200, fill="green")
        self.dino_y = 150  # Track dino's y position
        self.game_running = True

        # Queue for face-tracking communication
        self.queue = queue.Queue()

        # Start the face-tracking process in a separate thread
        self.face_tracking_thread = threading.Thread(target=self.run_face_tracking, daemon=True)
        self.face_tracking_thread.start()

        # Start game loop
        self.game_loop()

        # Start note generation (based on song array) every second
        self.root.after(1000, self.generate_note)

    def game_loop(self):
        """Main game loop."""
        if self.game_running:
            self.update_dino_position()
            self.move_note()
            self.check_collision()
            self.canvas.after(20, self.game_loop)  # 50 FPS

    def run_face_tracking(self):
        """Run the face tracking process and capture its output."""
        self.face_tracking_process = subprocess.Popen(
            ["python", "po.py"],  # Adjust filename if needed
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Ignore errors
            text=True,
            bufsize=1  # Line-buffered output for real-time streaming
        )

        if sys.platform == "win32":
            while self.game_running:
                output = self.face_tracking_process.stdout.readline().strip()
                if output:
                    try:
                        y_percentage = float(output)
                        self.queue.put(y_percentage)
                    except ValueError:
                        print(f"Invalid output from po.py: {output}")
                        # how to get length of array
        else:
            import select
            while self.game_running:
                ready_to_read, _, _ = select.select([self.face_tracking_process.stdout], [], [], 0.1)
                if ready_to_read:
                    output = self.face_tracking_process.stdout.readline().strip()
                    if output:
                        try:
                            y_percentage = float(output)
                            self.queue.put(y_percentage)
                        except ValueError:
                            print(f"Invalid output from po.py: {output}")

    def update_dino_position(self):
        """Update the position of the dino block on the canvas."""
        if not self.queue.empty():
            y_percentage = self.queue.get_nowait()
            self.dino_y = int(400 * (y_percentage / 100))  # Scale Y position
            self.canvas.coords(self.dino, 75, self.dino_y, 125, self.dino_y + 50)

    def generate_note(self):
        """Generate a new note based on the song array and current time."""
        global current_time, current_note

        # Check for the next note in the song array based on current_time
        for note in song:
            if note["height"] is not None and (current_time >= note["time_offset"]):
                if current_note is None:  # Only generate a new note if there's no current note
                    current_note = self.canvas.create_rectangle(
                        1200, note["height"], 1250, note["height"] + 50, fill="blue"
                    )
                    break  # Stop looking for more notes once one is created

        # Increment time
        current_time += 1
        self.root.after(1000, self.generate_note)

    def move_note(self):
        """Move the current note from right to left."""
        global current_note
        if current_note:
            x1, y1, x2, y2 = self.canvas.coords(current_note)
            if x2 > 0:
                self.canvas.move(current_note, -10, 0)  # Move left
            else:
                self.canvas.delete(current_note)
                current_note = None  # Remove note when off-screen

    def check_collision(self):
        """Check if the dino collides with the note."""
        if current_note:
            dino_coords = self.canvas.coords(self.dino)
            note_coords = self.canvas.coords(current_note)
            if self.overlapping(dino_coords, note_coords):
                self.game_over()

    def overlapping(self, rect1, rect2):
        """Check if two rectangles overlap."""
        x1, y1, x2, y2 = rect1
        ox1, oy1, ox2, oy2 = rect2
        return not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2)

    def game_over(self):
        """End the game and display a message."""
        self.game_running = False
        self.canvas.create_text(600, 200, text="Game Over", font=("Arial", 30), fill="red")

# Run the game
root = tk.Tk()
game = DinoGame(root)
root.mainloop()
