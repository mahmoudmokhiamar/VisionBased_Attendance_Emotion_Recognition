import cv2

def draw_overlay(frame, name=None, emotion=None, attendance_marked=False):
    """
    Draw name, emotion, and attendance status on the webcam frame.
    """
    overlay_text = ""
    
    # Append the name and emotion to the overlay text
    if name:
        overlay_text += f"Name: {name}  "
    if emotion:
        overlay_text += f"Emotion: {emotion}  "
    
    # Add attendance status if marked
    if attendance_marked:
        overlay_text += "✓ Marked Present"
    
    # Set position and font
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (10, 30)  # Adjust the starting position of the text
    font_scale = 0.8
    font_color = (0, 255, 0) if attendance_marked else (0, 0, 255)  # Green if marked, red otherwise
    thickness = 2
    
    # Draw text if there's any to display
    if overlay_text:
        cv2.putText(frame, overlay_text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    return frame
