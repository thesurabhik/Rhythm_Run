import tkinter as tk
from game import DinoGame

if __name__ == "__main__":
    root = tk.Tk()
    game = DinoGame(root)
    root.mainloop()