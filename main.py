import tkinter as tk
from game import DinoGame

if __name__ == "__main__":
    root = tk.Tk()  # Create the main window
    game = DinoGame(root)  # Initialize the game
    root.mainloop()  # Start the Tkinter main event loop