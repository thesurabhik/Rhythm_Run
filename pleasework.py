import tkinter as tk
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import threading

def start_dino_game():
    def run_game():
        # Set up the WebDriver (make sure to replace 'chromedriver' with the path to your WebDriver)
        driver = webdriver.Chrome()
        driver.get("chrome://dino")
        
        # Start the game by sending a space key
        body = driver.find_element("tag name", "body")
        body.send_keys(Keys.SPACE)
    
    # Run the game in a separate thread to avoid blocking the GUI
    threading.Thread(target=run_game, daemon=True).start()

# Create the GUI
root = tk.Tk()
root.title("Dinosaur Game Launcher")

# Add a button to start the game
start_button = tk.Button(root, text="Run Dinosaur Game", command=start_dino_game, font=("Arial", 16))
start_button.pack(pady=20)

# Run the GUI event loop
root.mainloop()