import pygame
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import platform

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

root = tk.Tk()
root.title('Hand Tracking with Resistance Value and Icon Dragging')

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

# Load icons
resistor_icon = pygame.image.load("Resistor-removebg-preview.png")
voltage_source_icon = pygame.image.load("dc_source-removebg-preview.png")
wire_icon = pygame.image.load("wire-removebg-preview.png")
capacitor_icon = pygame.image.load("capacitor-removebg-preview.png")
inductor_icon = pygame.image.load("inductor-removebg-preview.png")
current_source_icon = pygame.image.load("current_source-removebg-preview.png")

icon_size = (100, 100)
# Resize icons to a fixed size
resistor_icon = pygame.transform.scale(resistor_icon, icon_size)
voltage_source_icon = pygame.transform.scale(voltage_source_icon, icon_size)
wire_icon = pygame.transform.scale(wire_icon, icon_size)
capacitor_icon = pygame.transform.scale(capacitor_icon, icon_size)
inductor_icon = pygame.transform.scale(inductor_icon, icon_size)
current_source_icon = pygame.transform.scale(current_source_icon, icon_size)

# Initialize positions of icons
resistor_position = [500, 50]
voltage_source_position = [500, 150]
wire_position = [200, 50]
capacitor_position = [200, 150]
inductor_position = [350, 50]
current_source_position = [350, 150]

selected_icon = None
is_dragging = False
drag_start_position = (0, 0)

def update_frame(event=None):
    global is_dragging, selected_icon, drag_start_position, resistor_position, voltage_source_position, \
        wire_position, capacitor_position, inductor_position, current_source_position

    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    pygame_screen.fill(background_color)

    cursor_color = (0, 0, 255)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            cursor_position = (
                int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))
            pygame.draw.circle(pygame_screen, cursor_color, cursor_position, 10)

            if not is_dragging:
                if (resistor_position[0] < cursor_position[0] < resistor_position[0] + icon_size[0] and
                        resistor_position[1] < cursor_position[1] < resistor_position[1] + icon_size[1]):
                    selected_icon = 'resistor'
                elif (voltage_source_position[0] < cursor_position[0] < voltage_source_position[0] + icon_size[0] and
                        voltage_source_position[1] < cursor_position[1] < voltage_source_position[1] + icon_size[1]):
                    selected_icon = 'voltage_source'
                elif (wire_position[0] < cursor_position[0] < wire_position[0] + icon_size[0] and
                        wire_position[1] < cursor_position[1] < wire_position[1] + icon_size[1]):
                    selected_icon = 'wire'
                elif (capacitor_position[0] < cursor_position[0] < capacitor_position[0] + icon_size[0] and
                        capacitor_position[1] < cursor_position[1] < capacitor_position[1] + icon_size[1]):
                    selected_icon = 'capacitor'
                elif (inductor_position[0] < cursor_position[0] < inductor_position[0] + icon_size[0] and
                        inductor_position[1] < cursor_position[1] < inductor_position[1] + icon_size[1]):
                    selected_icon = 'inductor'
                elif (current_source_position[0] < cursor_position[0] < current_source_position[0] + icon_size[0] and
                        current_source_position[1] < cursor_position[1] < current_source_position[1] + icon_size[1]):
                    selected_icon = 'current_source'
                else:
                    selected_icon = None

            if selected_icon:
                is_dragging = True
                drag_start_position = cursor_position

    if is_dragging:
        if selected_icon == 'resistor':
            resistor_position = [drag_start_position[0] - icon_size[0] // 2, drag_start_position[1] - icon_size[1] // 2]
        elif selected_icon == 'voltage_source':
            voltage_source_position = [drag_start_position[0] - icon_size[0] // 2, drag_start_position[1] - icon_size[1] // 2]
        elif selected_icon == 'wire':
            wire_position = [drag_start_position[0] - icon_size[0] // 2, drag_start_position[1] - icon_size[1] // 2]
        elif selected_icon == 'capacitor':
            capacitor_position = [drag_start_position[0] - icon_size[0] // 2, drag_start_position[1] - icon_size[1] // 2]
        elif selected_icon == 'inductor':
            inductor_position = [drag_start_position[0] - icon_size[0] // 2, drag_start_position[1] - icon_size[1] // 2]
        elif selected_icon == 'current_source':
            current_source_position = [drag_start_position[0] - icon_size[0] // 2, drag_start_position[1] - icon_size[1] // 2]

        is_dragging = False

    # Draw icons at their positions
    pygame_screen.blit(resistor_icon, resistor_position)
    pygame_screen.blit(voltage_source_icon, voltage_source_position)
    pygame_screen.blit(wire_icon, wire_position)
    pygame_screen.blit(capacitor_icon, capacitor_position)
    pygame_screen.blit(inductor_icon, inductor_position)
    pygame_screen.blit(current_source_icon, current_source_position)

    # Draw grid dots
    dot_color = (0, 0, 0)
    dot_radius = 5
    for i in range(2):
        for j in range(4):
            dot_position = (400 + j * 100, 400 + i * 100)
            pygame.draw.circle(pygame_screen, dot_color, dot_position, dot_radius)

    pygame.display.flip()
    pygame.time.delay(10)
    root.after(10, update_frame)

update_frame()


# Start the Tkinter event loop
root.mainloop()

# Release resources when the application is closed
cap.release()
cv2.destroyAllWindows()
