from PIL import Image, ImageTk

class SpriteManager:
    def __init__(self):
        self.images = {}
        
    def load_images(self):
        """Load all game images or create placeholders if files don't exist."""
        try:
            # Try to load actual image files if they exist
            dino_image = Image.open("dino.png")
            self.images["dino"] = ImageTk.PhotoImage(dino_image)
            
            cactus_image = Image.open("cactus.png")
            self.images["cactus"] = ImageTk.PhotoImage(cactus_image)
            
            return True
        except FileNotFoundError:
            # If image files don't exist, create placeholder images
            print("Image files not found. Creating placeholder images.")
            self.create_placeholder_images()
            return False
    
    def create_placeholder_images(self):
        """Create placeholder images if real sprite files are missing."""
        # Create a simple dino placeholder (green rectangle with details)
        dino_placeholder = Image.new('RGBA', (50, 50), (0, 0, 0, 0))
        for y in range(50):
            for x in range(50):
                if 10 <= x <= 40 and 10 <= y <= 40:
                    # Body
                    dino_placeholder.putpixel((x, y), (0, 200, 0, 255))
                if 35 <= x <= 45 and 5 <= y <= 15:
                    # Head
                    dino_placeholder.putpixel((x, y), (0, 230, 0, 255))
                if x == 43 and y == 8:
                    # Eye
                    dino_placeholder.putpixel((x, y), (0, 0, 0, 255))
        
        self.images["dino"] = ImageTk.PhotoImage(dino_placeholder)
        
        # Create a simple cactus placeholder (red spiky shape)
        cactus_placeholder = Image.new('RGBA', (30, 40), (0, 0, 0, 0))
        for y in range(40):
            for x in range(30):
                if 10 <= x <= 20 and 0 <= y <= 40:
                    # Main stem
                    cactus_placeholder.putpixel((x, y), (200, 0, 0, 255))
                if ((0 <= x <= 10 and 10 <= y <= 20) or 
                    (20 <= x <= 30 and 15 <= y <= 25)):
                    # Arms
                    cactus_placeholder.putpixel((x, y), (220, 0, 0, 255))
        
        self.images["cactus"] = ImageTk.PhotoImage(cactus_placeholder)
    
    def get_image(self, name):
        """Get an image by name."""
        return self.images.get(name)