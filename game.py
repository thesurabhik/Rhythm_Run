import tkinter as tk
import threading
import subprocess
import queue
import time

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
        self.score = 0

        # Create a queue for communication between threads
        self.queue = queue.Queue()

        # Start the face-tracking process in a separate thread
        self.face_tracking_thread = threading.Thread(target=self.run_face_tracking)
        self.face_tracking_thread.daemon = True  # Ensure the thread exits when the main program exits
        self.face_tracking_thread.start()

        # Start game loop
        self.game_loop()

    def game_loop(self):
        """Main game loop."""
        if self.game_running:
            # Try to get a new y value from the queue
            self.update_dino_position()

            # Update the canvas (every 20ms for 50 FPS)
            self.canvas.after(20, self.game_loop)

    def run_face_tracking(self):
        """Run the face tracking process and capture its output."""
        # Start the face tracking script in a subprocess
        self.face_tracking_process = subprocess.Popen(
            ["python", "po.py"],  # Adjust filename if needed
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Continuously read output from the face tracking process
        while self.game_running:
            try:
                output = self.face_tracking_process.stdout.readline().strip()
                if output:
                    # Put the new y percentage into the queue
                    y_percentage = float(output)  # Convert string to float
                    self.queue.put(y_percentage)

            except Exception as e:
                print("Error in face tracking:", e)
                break

    def update_dino_position(self):
        """Update the position of the dino block on the canvas."""
        try:
            # Check if there is any new data in the queue
            if not self.queue.empty():
                y_percentage = self.queue.get_nowait()
                self.dino_y = int(50 + (y_percentage / 100) * 100)  # Scale Y position
                # Update the dino's position on the canvas
                self.canvas.coords(self.dino, 75, self.dino_y, 125, self.dino_y + 50)
        except queue.Empty:
            pass  # No new data in the queue

# Run the game
root = tk.Tk()
game = DinoGame(root)
root.mainloop()
