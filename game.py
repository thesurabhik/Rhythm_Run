import tkinter as tk
import threading
import subprocess
import queue
import sys
import os

class DinoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Dinosaur Game")

        # Set the game window size
        self.canvas = tk.Canvas(self.root, width=600, height=200, bg="white")
        self.canvas.pack()

        # Load the Dino sprite
        self.dino = self.canvas.create_rectangle(75, 150, 125, 200, fill="green")  # Simple square dino
        self.dino_y = 150  # Track dino's y position
        self.game_running = True

        # Create a queue for communication between threads
        self.queue = queue.Queue()

        # Start the face-tracking process in a separate thread
        self.face_tracking_thread = threading.Thread(target=self.run_face_tracking, daemon=True)
        self.face_tracking_thread.start()

        # Start game loop
        self.game_loop()

    def game_loop(self):
        """Main game loop."""
        if self.game_running:
            self.update_dino_position()
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
            # Windows: Use a simple readline loop
            while self.game_running:
                output = self.face_tracking_process.stdout.readline().strip()
                if output:
                    try:
                        y_percentage = float(output)
                        self.queue.put(y_percentage)
                    except ValueError:
                        print(f"Invalid output from po.py: {output}")
        else:
            # macOS/Linux: Use select.select()
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
            self.dino_y = int(200* (y_percentage / 100))  # Scale Y position
            self.canvas.coords(self.dino, 75, self.dino_y, 125, self.dino_y + 50)

# Run the game
root = tk.Tk()
game = DinoGame(root)
root.mainloop()
