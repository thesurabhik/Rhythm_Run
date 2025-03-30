import tkinter as tk
import threading
import subprocess
import queue
import sys
import os
import random


current = 0
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

        # List to store obstacles
        self.obstacles = []

        # Start the face-tracking process in a separate thread
        self.face_tracking_thread = threading.Thread(target=self.run_face_tracking, daemon=True)
        self.face_tracking_thread.start()

        # Start obstacle generation every 2 seconds
        self.root.after(2000, self.create_obstacle)
        
        # Start game loop
        self.game_loop()

    def game_loop(self):
        """Main game loop."""
        if self.game_running:
            self.update_dino_position()
            self.move_obstacles()
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

    def create_obstacle(self):
        """Create a new obstacle at a random height."""
        if self.game_running:
            #for note in song:
            note = song[current]
            note_time = note["time_offset"]  # Get note's time offset in seconds
            
            # Get the actual elapsed time
            current_time = time.time() - start_time

            # Stop when the song is done
            if (note_time >= current_time - 0.2 and note_time < current_time + 0.2):
                print(f"{current_time:.2f}s - Test completed.")
                break
            else:
                print(f"{current_time:.2f}s - {note['height']}")

            y_position = random.randint(50, 350)  # Random height
            obstacle = self.canvas.create_rectangle(1200, y_position, 1250, y_position + 50, fill="black")
            self.obstacles.append(obstacle)
            self.root.after(2000, self.create_obstacle)  # Repeat every 2 seconds

    def move_obstacles(self):
        """Move obstacles from right to left."""
        for obstacle in self.obstacles:
            x1, y1, x2, y2 = self.canvas.coords(obstacle)
            if x2 > 0:
                self.canvas.move(obstacle, -10, 0)  # Move left
            else:
                self.canvas.delete(obstacle)
                self.obstacles.remove(obstacle)  # Remove off-screen obstacles

    def check_collision(self):
        """Check if the dino collides with any obstacle."""
        dino_coords = self.canvas.coords(self.dino)
        for obstacle in self.obstacles:
            obs_coords = self.canvas.coords(obstacle)
            if self.overlapping(dino_coords, obs_coords):
                self.game_over()
                break

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
