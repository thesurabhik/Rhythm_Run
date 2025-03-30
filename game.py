import tkinter as tk
import random
from sprites import SpriteManager
from physics import PhysicsEngine

class DinoGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Dinosaur Game")

        # Set the game window size
        self.canvas = tk.Canvas(self.root, width=600, height=200, bg="white")
        self.canvas.pack()

        # Load sprites
        self.sprite_manager = SpriteManager()
        self.sprite_manager.load_images()
        
        # Initialize physics engine
        self.physics = PhysicsEngine()
        
        # Create the dino sprite
        self.dino = self.canvas.create_image(
            75, 150, 
            image=self.sprite_manager.get_image("dino"), 
            anchor="nw"
        )
        
        # Set initial values
        self.dino_y = 150  # Track the dino's y position
        self.jump_velocity = 0
        self.is_jumping = False
        
        # Game state
        self.game_running = True
        self.score = 0
        self.score_text = self.canvas.create_text(
            500, 30, 
            text=f"Score: {self.score}", 
            font=("Arial", 12)
        )

        # Start the game loop
        self.game_loop()

        # Bind the space key to make the dino jump
        self.root.bind("<space>", self.jump)

    def game_loop(self):
        """Main game loop."""
        if self.game_running:
            # Apply physics to dino
            self.apply_physics()
            
            # Move obstacles
            self.move_obstacles()
            
            # Update score
            self.score += 1
            self.canvas.itemconfig(self.score_text, text=f"Score: {self.score}")

            # Update the canvas
            self.canvas.after(20, self.game_loop)

    def apply_physics(self):
        """Apply physics to the dinosaur."""
        new_y, new_velocity, is_jumping = self.physics.apply_gravity(
            self.dino_y, self.jump_velocity, self.is_jumping
        )
        
        # Apply movement
        movement = new_y - self.dino_y
        self.canvas.move(self.dino, 0, movement)
        
        # Update state
        self.dino_y = new_y
        self.jump_velocity = new_velocity
        self.is_jumping = is_jumping

    def move_obstacles(self):
        """Move the obstacles from right to left."""
        # Add a new obstacle if one is not on the screen
        if random.randint(1, 50) == 1:
            self.canvas.create_image(
                600, 150, 
                image=self.sprite_manager.get_image("cactus"), 
                tag="obstacle", 
                anchor="nw"
            )

        # Move obstacles
        for obstacle in self.canvas.find_withtag("obstacle"):
            self.canvas.move(obstacle, -5, 0)
            if self.canvas.coords(obstacle)[0] < -30:  # If obstacle is off screen
                self.canvas.delete(obstacle)

        # Check for collision
        for obstacle in self.canvas.find_withtag("obstacle"):
            if self.check_collision(obstacle):
                self.game_over()
                return

    def jump(self, event):
        """Handle the jump action."""
        if not self.is_jumping:
            self.is_jumping = True
            self.jump_velocity = self.physics.initiate_jump()

    def check_collision(self, obstacle):
        """Check if the dino collides with an obstacle."""
        dino_coords = self.canvas.bbox(self.dino)
        obstacle_coords = self.canvas.bbox(obstacle)
        
        if dino_coords and obstacle_coords:  # Make sure both exist
            return self.physics.check_collision(dino_coords, obstacle_coords)
        return False

    def game_over(self):
        """Display the 'Game Over' message."""
        self.game_running = False
        self.canvas.create_text(
            300, 100, 
            text=f"Game Over! Score: {self.score}", 
            font=("Arial", 24), 
            fill="red"
        )
        
        # Add restart button
        restart_button = tk.Button(self.root, text="Restart", command=self.restart_game)
        restart_button_window = self.canvas.create_window(300, 130, window=restart_button)

    def restart_game(self):
        """Restart the game."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Create the dino sprite again
        self.dino = self.canvas.create_image(
            75, 150, 
            image=self.sprite_manager.get_image("dino"), 
            anchor="nw"
        )
        
        # Reset variables
        self.dino_y = 150
        self.jump_velocity = 0
        self.is_jumping = False
        self.game_running = True
        self.score = 0
        
        # Recreate score text
        self.score_text = self.canvas.create_text(
            500, 30, 
            text=f"Score: {self.score}", 
            font=("Arial", 12)
        )
        
        # Restart game loop
        self.game_loop()