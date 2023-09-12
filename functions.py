import numpy as np

# Average color function (pass in box) (returns BGR)
def get_average_color(a):
    avg_color = np.mean(a, axis=(0,1))
    return avg_color

def euclidean_distance(color1, color2):
    # Calculate the Euclidean distance between two colors
    return np.sqrt(np.sum((color1 - color2) ** 2))

def assign_custom_label(color, team1_bgr, team2_bgr):

    # Define the threshold distance for outliers
    threshold_distance = 120.0  # Adjust this value as needed based on the color space and your application

    # Calculate the distance to team1 and team2 colors
    team1_distance = euclidean_distance(color, team1_bgr)
    team2_distance = euclidean_distance(color, team2_bgr)

    # Check if the color is too far from both team2 and team1
    if team1_distance > threshold_distance and team2_distance > threshold_distance:
        return "group3"
    elif team1_distance < team2_distance:
        return "group1"
    else:
        return "group2"

def classify_bgr_color(bgr_color, team1_bgr, team2_bgr):
    # Convert BGR color to numpy array
    bgr_color = np.array(bgr_color)
    
    # Assign custom label based on color
    label = assign_custom_label(bgr_color, team1_bgr, team2_bgr)
    
    return label