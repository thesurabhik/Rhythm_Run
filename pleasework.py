import tkinter as tk
import random

class DinoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Dinosaur Game")

        # Set the game window size
        self.canvas = tk.Canvas(self.root, width=600, height=200, bg="white")
        self.canvas.pack()

        # Create the Dino (a rectangle for simplicity)
        self.dino = self.canvas.create_rectangle(50, 150, 100, 170, fill="green")

        # Set initial values
        self.is_jumping = False
        self.jump_height = 0

        # Start the game loop
        self.game_loop()

        # Bind the space key to make the dino jump
        self.root.bind("<space>", self.jump)

    def game_loop(self):
        """Main game loop."""
        # Move obstacles (for simplicity, we make them just rectangles)
        self.move_obstacles()

        # Update the canvas
        self.canvas.after(20, self.game_loop)

    def move_obstacles(self):
        """Move the obstacles from right to left."""
        # Add a new obstacle if one is not on the screen
        if random.randint(1, 50) == 1:
            self.canvas.create_rectangle(600, 150, 650, 170, fill="red", tag="obstacle")

        # Move obstacles
        for obstacle in self.canvas.find_withtag("obstacle"):
            self.canvas.move(obstacle, -5, 0)
            if self.canvas.coords(obstacle)[2] < 0:  # If obstacle is off screen
                self.canvas.delete(obstacle)

        # Check for collision (basic check)
        for obstacle in self.canvas.find_withtag("obstacle"):
            if self.check_collision(obstacle):
                self.game_over()
                return

    def jump(self, event):
        """Handle the jump action."""
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_height = 0
            self.animate_jump()

    def animate_jump(self):
        """Animate the jump action."""
        if self.is_jumping:
            # Move the dino up and down (simplified jump animation)
            if self.jump_height < 20:
                self.canvas.move(self.dino, 0, -5)
                self.jump_height += 5
                self.canvas.after(20, self.animate_jump)
            elif self.jump_height < 40:
                self.canvas.move(self.dino, 0, 5)
                self.jump_height += 5
                self.canvas.after(20, self.animate_jump)
            else:
                self.is_jumping = False

    def check_collision(self, obstacle):
        """Check if the dino collides with an obstacle."""
        dino_coords = self.canvas.coords(self.dino)
        obstacle_coords = self.canvas.coords(obstacle)
        
        # Simple collision check (if dino and obstacle overlap)
        if (dino_coords[2] > obstacle_coords[0] and dino_coords[0] < obstacle_coords[2] and
            dino_coords[3] > obstacle_coords[1] and dino_coords[1] < obstacle_coords[3]):
            return True
        return False

    def game_over(self):
        """Display the 'Game Over' message."""
        self.canvas.create_text(300, 100, text="Game Over!", font=("Arial", 24), fill="red")


if __name__ == "__main__":
    root = tk.Tk()
    game = DinoGame(root)
    root.mainloop()
