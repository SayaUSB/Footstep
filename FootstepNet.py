from Capsules.Position import Position
from Capsules.FootstepPlanner import FootstepPlanner
import matplotlib.pyplot as plt
import numpy as np

def print_footstep(step):
    foot = "Left" if step.is_left_foot else "Right"
    print(f"{foot} footat ({step.position.x:.2f}, {step.position.y:.2f}), orientation: {step.position.theta:.2f}")

def plot_footsteps(start, goal, path, obstacles):
    plt.figure(figsize=(10, 6))
    plt.title("Footstep Path")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot(start.x, start.y, "go", markersize=10, label="Start")
    plt.plot(goal.x, goal.y, "ro", markersize=10, label="Goal")

    # Plot obstacles
    for obstacle in obstacles:
        plt.plot(obstacle.x, obstacle.y, 'kx', markersize=10)
        circle = plt.Circle((obstacle.x, obstacle.y), 0.3, color="gray", fill=False)
        plt.gca().add_artist(circle)

    for i, step in enumerate(path):
        color = 'blue' if step.is_left_foot else 'orange'
        marker = 'o' if step.is_left_foot else 's'
        plt.plot(step.position.x, step.position.y, color=color, marker=marker, markersize=8)
        plt.text(step.position.x, step.position.y, str(i+1), fontsize=8, ha='center', va='center')

        # Plot orientation
        orientation_length = 0.1
        dx = orientation_length * np.cos(step.position.theta)
        dy = orientation_length * np.sin(step.position.theta)
        plt.arrow(step.position.x, step.position.y, dx, dy, head_width=0.05, head_length=0.05, fc=color, ec=color)

    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    planner = FootstepPlanner(step_length=0.3, step_width=0.2)
    start = Position(0, 0, 0)
    goal = Position(10, 0, 0)

    obstacles = [
        Position(5, 0, 0),
        Position(5, 0.55, 0),
        Position(5, -0.55, 0),
        Position(7, 0, 0),
        Position(8, 0, 0),
        Position(9, 0, 0),
        Position(9.5, 0.25, 0),
        Position(9.5, -0.25, 0)
    ]

    planner.set_obstacles(obstacles)
    path = planner.plan_path(start, goal)

    # print(f"Footstepplan from ({start.x}, {start.y}) to ({goal.x}, {goal.y}):")
    # for step in path:
    #     print_footstep(step)

    plot_footsteps(start, goal, path, obstacles)