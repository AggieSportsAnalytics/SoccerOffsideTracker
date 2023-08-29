import numpy as np
from sklearn.cluster import KMeans


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

# def euclidean_distance(color1, color2):
#     # Calculate the Euclidean distance between two colors
#     return np.sqrt(np.sum((color1 - color2) ** 2))

# def assign_custom_labels(bgr_array, team1_bgr, team2_bgr):

#     # Initialize an array to store the labels
#     labels = np.zeros(len(bgr_array), dtype=int)

#     # Define the threshold distance for outliers
#     threshold_distance = 120.0  # Adjust this value as needed based on the color space and your application

#     for i, color in enumerate(bgr_array):
#         # Calculate the distance to team1 and team2 colors
#         team1_distance = euclidean_distance(color, team1_bgr)
#         team2_distance = euclidean_distance(color, team2_bgr)

#         # Check if the color is too far from both team2 and team1
#         if team1_distance > threshold_distance and team2_distance > threshold_distance:
#             labels[i] = 3
#         elif team1_distance < team2_distance:
#             labels[i] = 1
#         else:
#             labels[i] = 2

#     return labels

# def divide_colors(bgr_array, team1_bgr, team2_bgr):
#     # Convert BGR array to numpy array
#     bgr_array = np.array(bgr_array)

#     # Assign custom labels based on colors
#     labels = assign_custom_labels(bgr_array, team1_bgr, team2_bgr)

#     # Separate the BGR values based on the assigned labels
#     group1_indices = np.where(labels == 1)[0]
#     group2_indices = np.where(labels == 2)[0]
#     group3_indices = np.where(labels == 3)[0]  # Colors that are too far from team2 or team1 are labeled as group 3

#     # Get the BGR values and indices for each group
#     group1_colors = bgr_array[group1_indices]
#     group1_indices_orig = group1_indices.tolist()

#     group2_colors = bgr_array[group2_indices]
#     group2_indices_orig = group2_indices.tolist()

#     group3_colors = bgr_array[group3_indices]
#     group3_indices_orig = group3_indices.tolist()

#     return group1_colors, group1_indices_orig, group2_colors, group2_indices_orig, group3_colors, group3_indices_orig