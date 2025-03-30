import tkinter as tk
import threading
import subprocess
import queue
import sys
import os
import pygame
import time



#stop breaking
#fix
LOW = "LOW"
MID = "MID"
HIGH = "HIGH"
NONE = "NONE"
DONE = "DONE"  # Marker for end of song

# Hardcoded song with time offsets in seconds
song = [
    {"height": NONE, "time_offset": 0},  # No note at start
    {"height": LOW, "time_offset": 6},  # Note at 6s
    {"height": MID, "time_offset": 8},  # Note at 8s
    {"height": HIGH, "time_offset": 10},  # Note at 10
    {"height": LOW, "time_offset": 12},  # Note at 12
    {"height": MID, "time_offset": 14},  # Note at 14
    {"height": HIGH, "time_offset": 16},  # Note at 16
    {"height": LOW, "time_offset": 18},  # Note at 18
    {"height": HIGH, "time_offset": 20},  # Note at 20
    {"height": LOW, "time_offset": 22},  # Note at 20 TEMPO SWITCH
    {"height": MID, "time_offset": 23},  # Note at 20
    {"height": HIGH, "time_offset": 24},  # Note at 20
    {"height": HIGH, "time_offset": 25},  # Note at 20
    {"height": LOW, "time_offset": 26},  # Note at 20
    {"height": HIGH, "time_offset": 27},  # Note at 20
    {"height": LOW, "time_offset": 28},  # Note at 20
    {"height": HIGH, "time_offset": 29},  # Note at 30
    {"height": MID, "time_offset": 30},  # Note at 32
    {"height": LOW, "time_offset": 31},  # Note at 34
    {"height": LOW, "time_offset": 32},  # Note at 34
    {"height": MID, "time_offset": 33},  # Note at 36
    {"height": HIGH, "time_offset": 34},  # Note at 38 ALL
    {"height": LOW, "time_offset": 35},  # Note at 40
    {"height": HIGH, "time_offset": 36},  # Note at 42
    {"height": LOW, "time_offset": 37},  # Note at 44
    {"height": HIGH, "time_offset": 38},  # Note at 46
    {"height": LOW, "time_offset": 39},   # End of song marker
    {"height": HIGH, "time_offset": 40},  # Note at 34
    {"height": MID, "time_offset": 41},  # Note at 34
    {"height": HIGH, "time_offset": 42},  # Note at 34 DANCE
    {"height": LOW, "time_offset": 43},  # Note at 34
    {"height": MID, "time_offset": 44},  # Note at 34
    {"height": HIGH, "time_offset": 45},  # Note at 34
    {"height": LOW, "time_offset": 46},  # Note at 34
    {"height": MID, "time_offset": 47},  # Note at 34
    {"height": HIGH, "time_offset": 48},  # Note at 34
    {"height": LOW, "time_offset": 49},   # End of song marker
    {"height": HIGH, "time_offset": 50},  # YOU
    {"height": HIGH, "time_offset": 51},  # MAKE
    {"height": HIGH, "time_offset": 52},  # ME
    {"height": LOW, "time_offset": 53},  # FEEL LIKE I'M
    {"height": HIGH, "time_offset": 54},  # TEEN
    {"height": HIGH, "time_offset": 55},  # AGE
    {"height": HIGH, "time_offset": 56},  # DREAM
    {"height": LOW, "time_offset": 57},  # THE WAY
    {"height": HIGH, "time_offset": 58},  # I
    {"height": HIGH, "time_offset": 59},  # CANT
    {"height": HIGH, "time_offset": 60},  # WAIT
    {"height": LOW, "time_offset": 61},  # LET'S RUN AWAY
    {"height": MID, "time_offset": 62},  # AND DONT
    {"height": LOW, "time_offset": 63},  # EVER LOOK
    {"height": HIGH, "time_offset": 64},  # BACK
    {"height": MID, "time_offset": 65},  # AND DONT
    {"height": LOW, "time_offset": 66},  # EVER LOOK
    {"height": NONE, "time_offset":67}, 
    {"height": DONE, "time_offset": 68}, 

]




class DinoGame:
    def play_start_song(self):
        #print("Playing start song...")
        pygame.mixer.init()
        pygame.mixer.music.load("TeenageDream.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pass
    def __init__(self, root):
        self.root = root
        self.root.title("Rhythm Run")
        self.current_index = 1
        self.score = 0
        

        # Set the game window size
        self.canvas = tk.Canvas(self.root, width=1200, height=400, bg="white")
        self.canvas.pack()

        # Create the Dino (Green Square)
        self.dino = self.canvas.create_rectangle(75, 150, 125, 200, fill="green")
        self.dino_y = 150  # Track dino's y position
        self.game_running = True
        
        self.score_text = self.canvas.create_text(900, 100, text=f"Score: {self.score}", font=("Arial", 30), fill="black")
        # Queue for face-tracking communication
        self.queue = queue.Queue()

        # List to store obstacles
        self.obstacles = []

        self.start_time = time.time()

        # Start the face-tracking process in a separate thread
        self.face_tracking_thread = threading.Thread(target=self.run_face_tracking, daemon=True)
        self.face_tracking_thread.start()

        # Start obstacle generation every 2 seconds
        self.root.after(2000, self.create_obstacle)
        
        # Start game loop
        self.music_thread = threading.Thread(target=self.play_start_song)
        self.music_thread.start()
        self.game_loop()

    def game_loop(self):
        """Main game loop."""
        if self.game_running:
            self.update_dino_position()
            self.create_obstacle()
            self.move_obstacles()
            self.check_collision()
            self.canvas.after(20, self.game_loop)  # 50 FPS
        else:
            self.face_tracking_process.terminate()
            self.face_tracking_process.wait()
            self.second_round = subprocess.Popen(["python", "gumpa.py"])

        

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
            
            
                note = song[self.current_index]
                note_time = note["time_offset"]  # Get note's time offset in seconds
                
                # Get the actual elapsed time
                current_time = time.time() - self.start_time

                # Stop when the song is done
                if (note_time >= current_time and note_time < current_time + 0.1):
                    print("testing")
                    y_position = -100
                    if(note["height"] == NONE):
                        y_position = -100 
                    if(note["height"] == DONE):
                        self.game_over()
                    if(note["height"] == HIGH):
                        y_position = 100  # Random height
                    if(note["height"] == MID):
                        y_position = 200
                    if(note["height"] == LOW):
                        y_position = 300
                    obstacle = self.canvas.create_rectangle(1200, y_position, 1250, y_position + 50, fill="black")
                    self.obstacles.append(obstacle)
                    #self.root.after(2000, self.create_obstacle)  # Repeat every 2 seconds
                    self.current_index += 1

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
                #self.game_over()
                #self.score = self.score + 1
                self.score += 1  # Increase score by 1 for each collision
                self.canvas.itemconfig(self.score_text, text=f"Score: {self.score}")
                self.canvas.delete(obstacle)
                self.obstacles.remove(obstacle)
                break

    def overlapping(self, rect1, rect2):
        """Check if two rectangles overlap."""
        x1, y1, x2, y2 = rect1
        ox1, oy1, ox2, oy2 = rect2
        return not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2)
    

    def game_over(self):
        """End the game and display a message."""
        pygame.mixer.music.stop()
        self.game_running = False
        self.canvas.create_text(600, 200, text="Game Over", font=("Arial", 30), fill="red")

# Run the game
root = tk.Tk()
game = DinoGame(root)
root.mainloop()