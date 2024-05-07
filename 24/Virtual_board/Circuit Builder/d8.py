import pygame
import numpy as np
import mediapipe as mp
import cv2
import platform

# Initialize Pygame
pygame.init()

# Set up Pygame display
screen_width, screen_height = 800, 600  # Default resolution if platform is not macOS
if platform.system() == 'Darwin':
    from AppKit import NSScreen
    screen_width = int(NSScreen.mainScreen().frame().size.width)
    screen_height = int(NSScreen.mainScreen().frame().size.height)
pygame_screen = pygame.display.set_mode((screen_width, screen_height))

background_color = (255, 255, 255)

# Initialize variables for drawing lines
line_color = (0, 0, 0)
line_thickness = 5
line_points = []

# Initialize variables for hand tracking
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize grid properties
grid_spacing = 100
grid_origin = (screen_width // 2 - grid_spacing * 2, screen_height // 2 - grid_spacing)
grid_dots = [(grid_origin[0] + j * grid_spacing, grid_origin[1] + i * grid_spacing) for i in range(2) for j in range(4)]

def draw_grid():
    # Draw grid dots
    dot_color = (0, 0, 0)
    dot_radius = 5
    for dot_position in grid_dots:
        pygame.draw.circle(pygame_screen, dot_color, dot_position, dot_radius)

def update_frame():
    global line_points

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        pygame_screen.fill(background_color)

        draw_grid()

        index_finger_position = None  # Initialize outside the loop

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_position = (
                    int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))

                # Check if index finger is within the grid region
                if (grid_origin[0] <= index_finger_position[0] <= grid_origin[0] + grid_spacing * 4 and
                    grid_origin[1] <= index_finger_position[1] <= grid_origin[1] + grid_spacing * 2):

                    # Find the closest dot to the index finger
                    closest_dot = min(grid_dots, key=lambda dot: np.linalg.norm(np.array(index_finger_position) - np.array(dot)))

                    # Add index finger position to line points if it's not already there
                    if closest_dot not in line_points:
                        line_points.append(closest_dot)

        # Draw lines between consecutive points in line_points
        if len(line_points) > 1:
            for i in range(len(line_points) - 1):
                pygame.draw.line(pygame_screen, line_color, line_points[i], line_points[i + 1], line_thickness)

        # Draw pointer if index finger position is available
        if index_finger_position is not None:
            pygame.draw.circle(pygame_screen, (0, 0, 255), index_finger_position, 10)

        # Check if backspace key is pressed to delete the last drawn line
        keys = pygame.key.get_pressed()
        if keys[pygame.K_BACKSPACE]:
            if line_points:
                line_points.pop()

        pygame.display.flip()
        pygame.time.delay(10)
        pygame.event.pump()
        pygame.display.update()

update_frame()

# Release resources when the application is closed
cap.release()
cv2.destroyAllWindows()
