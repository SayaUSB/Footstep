import numpy as np
from .Footstep import Footstep
from .Position import Position

class FootstepPlanner:
    def __init__(self, step_length, step_width, safety_distance=1.0):
        self.step_length = step_length
        self.step_width = step_width
        self.safety_distance = safety_distance
        self.obstacles = []

    def set_obstacles(self, obstacles):
        self.obstacles = obstacles

    def plan_path(self,start, goal):
        path = []
        current_pos = start
        is_left_foot = True
        while self.calculate_distance(current_pos, goal) > self.step_length:
            next_step = self.take_step(current_pos, goal, is_left_foot)
            adjusted_step = self.adjust_step(next_step)
            if adjusted_step:
                path.append(adjusted_step)
                current_pos = adjusted_step.position
                is_left_foot = not is_left_foot
            else:
                alternative_step = self.find_alternative_step(current_pos, goal, is_left_foot)
                if alternative_step:
                    path.append(alternative_step)
                    current_pos = alternative_step.position
                    is_left_foot = not is_left_foot
                else:
                    sidestep = self.find_sidestep(current_pos, is_left_foot)
                    if sidestep:
                        path.append(sidestep)
                        current_pos = sidestep.position
                        is_left_foot = not is_left_foot
                    else:
                        print("No valid step found!")
                        return path
        final_step = Footstep(goal, is_left_foot)
        path.append(final_step)
        return path
    
    def adjust_step(self, step):
        if self.is_valid_step(step):
            return step
        
        angles = np.linspace(-np.pi/2, np.pi/2, 18)
        for angle in angles:
            adjusted_pos = self.rotate_point(step.position, Position(step.position.x, step.position.y, 0), angle)
            if adjusted_pos:
                adjusted_step = Footstep(adjusted_pos, step.is_left_foot)
                if self.is_valid_step(adjusted_step):
                    return adjusted_step
        return None
    
    def is_valid_step(self, step):
        if step is None:
            return False
        for obstacle in self.obstacles:
            if self.calculate_distance(step.position, obstacle) < self.step_length+0.126:
                return False
        return True
    
    def find_alternative_step(self, current_pos, goal, is_left_foot):
        angles = np.linspace(-np.pi/2, np.pi/2, 18)
        for angle in angles:
            rotated_pos = self.rotate_point(current_pos, goal, angle)
            if rotated_pos:
                step = self.take_step(current_pos, rotated_pos, is_left_foot)
                if self.is_valid_step(step):
                    return step
        return None
    
    def take_step(self, current_pos, goal, is_left_foot):
        dx = goal.x - current_pos.x
        dy = goal.y - current_pos.y
        distance = np.sqrt(dx**2 + dy**2)

        step_x = round(dx/distance, 2) * self.step_length
        step_y = round(dy/distance, 2) * self.step_length

        new_x = current_pos.x + step_x
        new_y = current_pos.y + step_y
        new_theta = np.arctan2(step_y, step_x)

        if not (np.isfinite(new_x) and np.isfinite(new_y) and np.isfinite(new_theta)):
            return None
        
        new_pos = Position(new_x, new_y, new_theta)

        lateral_offset = self.step_width/2 if is_left_foot else -self.step_width/2
        new_pos.x += -np.sin(new_pos.theta) * lateral_offset
        new_pos.y += np.cos(new_pos.theta) * lateral_offset

        if not (np.isfinite(new_pos.x) and np.isfinite(new_pos.y)):
            return None

        return Footstep(new_pos, is_left_foot)
    
    def rotate_point(self, point, origin, angle):
        ox, oy = origin.x, origin.y
        px, py = point.x, point.y
        qx = ox + np.cos(angle)*(px - ox) - np.sin(angle)*(py - oy)
        qy = oy + np.sin(angle)*(px - ox) + np.cos(angle)*(py - oy)
        if not (np.isfinite(qx) and np.isfinite(qy)):
            return None
        return Position(qx, qy, point.theta)

    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
    
    def find_sidestep(self, current_pos, is_left_foot):
        side_angles = [np.pi/2, -np.pi/2]
        for angle in side_angles:
            rotated_pos = self.rotate_point(Position(current_pos.x + self.step_length, current_pos.y, current_pos.theta ), current_pos, angle)
            if rotated_pos:
                step = self.take_step(current_pos, rotated_pos, is_left_foot)
                if self.is_valid_step(step):
                    return step
        return None