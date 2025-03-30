class PhysicsEngine:
    def __init__(self):
        self.gravity = 0.6
        self.jump_strength = -12  # Negative because y-axis is inverted in tkinter
        self.ground_level = 150
    
    def apply_gravity(self, current_y, velocity, is_jumping):
        """Apply gravity and calculate new position."""
        # Apply gravity to velocity
        new_velocity = velocity + self.gravity
        
        # Calculate new position
        new_y = current_y + new_velocity
        
        # Check if object has landed
        if new_y >= self.ground_level:
            new_y = self.ground_level
            new_velocity = 0
            is_jumping = False
        
        return new_y, new_velocity, is_jumping
    
    def initiate_jump(self):
        """Get initial jump velocity."""
        return self.jump_strength
    
    def check_collision(self, object1_box, object2_box):
        """Check if two bounding boxes collide."""
        # Make collision boxes slightly smaller for better gameplay
        obj1 = (object1_box[0] + 10, object1_box[1] + 10, 
                object1_box[2] - 10, object1_box[3] - 5)
        obj2 = (object2_box[0] + 5, object2_box[1] + 5, 
                object2_box[2] - 5, object2_box[3] - 5)
        
        # Check if the boxes overlap
        if (obj1[2] > obj2[0] and obj1[0] < obj2[2] and
            obj1[3] > obj2[1] and obj1[1] < obj2[3]):
            return True
        return False