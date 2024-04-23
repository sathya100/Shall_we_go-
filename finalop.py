import tkinter as tk
from tkinter import messagebox
import requests
import json
import os
import urllib.request
from PIL import Image, ImageTk
import torch
import numpy as np
from PIL import Image
import requests
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import os
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib import cm
import tkinter as tk
import gmplot
import googlemaps
import math


# Initialize Google Maps API client
client = googlemaps.Client(key="AIzaSyCPK6NWKNZsxIsi7NuFVadReAlGPCPtcv4")  

# Dictionary to map colors to text tags
color_tags = {
    "origin": {"text_color": "black", "marker_color": "black"},
    "destination1": {"text_color": "blue", "marker_color": "blue"},
    "destination2": {"text_color": "green", "marker_color": "green"},
    "destination3": {"text_color": "purple", "marker_color": "purple"},
    "destination4": {"text_color": "orange", "marker_color": "orange"},
    "destination5": {"text_color": "brown", "marker_color": "brown"},
    "destination6": {"text_color": "pink", "marker_color": "pink"},
    "destination7": {"text_color": "cyan", "marker_color": "cyan"},
    "destination8": {"text_color": "magenta", "marker_color": "magenta"},
    "destination9": {"text_color": "yellow", "marker_color": "yellow"},
    "destination10": {"text_color": "teal", "marker_color": "teal"},
    "destination11": {"text_color": "lime", "marker_color": "lime"},
    "destination12": {"text_color": "olive", "marker_color": "olive"},
    "destination13": {"text_color": "maroon", "marker_color": "maroon"},
    "destination14": {"text_color": "navy", "marker_color": "navy"},
    "destination15": {"text_color": "indigo", "marker_color": "indigo"},
    "destination16": {"text_color": "slategray", "marker_color": "slategray"},
    "destination17": {"text_color": "orchid", "marker_color": "orchid"},
    "destination18": {"text_color": "skyblue", "marker_color": "skyblue"},
    "destination19": {"text_color": "coral", "marker_color": "coral"},
    "destination20": {"text_color": "sienna", "marker_color": "sienna"},
    # Add more destinations with different colors as needed
}



# Initialize model and processor
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

def MetaParse(MetaUrl, key):
    response = urllib.request.urlopen(MetaUrl)
    jsonRaw = response.read()
    jsonData = json.loads(jsonRaw)
    if jsonData['status'] == "OK":
        if 'date' in jsonData:
            return (jsonData['date'], jsonData['pano_id'])
        else:
            return (None, jsonData['pano_id'])
    else:
        return (None, None)

def GetStreetLL(Lat, Lon, File, SaveLoc, key):
    base = r"https://maps.googleapis.com/maps/api/streetview"
    size = r"?size=1200x800&fov=60&location="
    headings = range(0, 360, 45)
    
    # Remove existing files from the directory
    existing_files = os.listdir(SaveLoc)
    for file in existing_files:
        os.remove(os.path.join(SaveLoc, file))
    
    PrevImage = []
    for heading in headings:
        end = str(Lat) + "," + str(Lon) + "&heading=" + str(heading) + key
        MyUrl = base + size + end
        fi = f"{File}_{heading}.jpg"
        MetaUrl = base + r"/metadata" + size + end
        met_lis = list(MetaParse(MetaUrl, key))
        if (met_lis[1], heading) not in PrevImage and met_lis[0] is not None:
            urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc, fi))
            met_lis.append(fi)
            PrevImage.append((met_lis[1], heading))
        else:
            met_lis.append(None)
    return "Images saved successfully"

def fetch_air_quality(latitude, longitude):
    url = 'https://airquality.googleapis.com/v1/history:lookup?key=AIzaSyCPK6NWKNZsxIsi7NuFVadReAlGPCPtcv4'  # Replace YOUR_API_KEY with your actual API key

    data = {
        "dateTime": "2024-03-21T15:01:23Z",
        "location": {
            "latitude": latitude,
            "longitude": longitude
        }
    }

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, data=json.dumps(data), headers=headers)

    response_data = response.json()
    air_quality_info = ""
    hours_info = response_data.get('hoursInfo', [])
    for hour_info in hours_info:
        air_quality_info += f"DateTime: {hour_info.get('dateTime')}\n"
        indexes = hour_info.get('indexes', [])
        for index in indexes:
            air_quality_info += f"Index Code: {index.get('code')}\n"
            air_quality_info += f"Display Name: {index.get('displayName')}\n"
            air_quality_info += f"AQI: {index.get('aqi')}\n"
            air_quality_info += f"AQI Display: {index.get('aqiDisplay')}\n"
            color = index.get('color', {})
            air_quality_info += f"Color (RGB): {color}\n"
            air_quality_info += f"Category: {index.get('category')}\n"
            air_quality_info += f"Dominant Pollutant: {index.get('dominantPollutant')}\n\n"

    return air_quality_info

def on_street_view_click():
    try:
        latitude = float(latitude_entry.get())
        longitude = float(longitude_entry.get())
        file_name = file_name_entry.get()
        save_location = save_location_entry.get()
        key = "&key=AIzaSyCPK6NWKNZsxIsi7NuFVadReAlGPCPtcv4"  # Replace YOUR_API_KEY with your actual API key

        street_view_result = GetStreetLL(latitude, longitude, file_name, save_location, key)

        messagebox.showinfo("Street View Result", street_view_result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def on_air_quality_click():
    try:
        latitude = float(latitude_entry.get())
        longitude = float(longitude_entry.get())

        air_quality_result = fetch_air_quality(latitude, longitude)

        messagebox.showinfo("Air Quality Result", air_quality_result)
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Calculate average GVI

processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-panoptic")

# Default folder path
folder_path = r"C:\Users\sthdh\OneDrive\Desktop\gvi\db"

def draw_panoptic_segmentation(segmentation, segments_info):
    # Get the used color map
    viridis = cm.get_cmap('viridis', torch.max(segmentation))
    fig, ax = plt.subplots()
    ax.imshow(segmentation)
    instances_counter = defaultdict(int)
    handles = []
    labels = []  # Store labels in an array
    segmentation_mask = np.zeros_like(segmentation)
    # For each segment, draw its legend and create segmentation mask
    for segment in segments_info:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id]}"
        instances_counter[segment_label_id] += 1
        color = viridis(segment_id)
        handles.append(mpatches.Patch(color=color, label=label))
        labels.append(label)  # Store label in the array
        segmentation_mask[segmentation == segment_id] = segment_id

    ax.legend(handles=handles)
    return labels, segmentation_mask  # Return the labels array and segmentation mask

def calculate_avg_gvi():
    total_gvi = 0
    num_images = 0

    # Iterate over each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)

            # Preprocess the image using the image processor
            inputs = processor(images=image, return_tensors="pt")

            # Forward pass through the model
            with torch.no_grad():
                outputs = model(**inputs)

            # Postprocess the outputs
            results = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

            # Visualize the panoptic segmentation
            labels, segmentation_mask = draw_panoptic_segmentation(**results)
            plt.show()

            # Calculate the number of pixels under the "tree" mask and total pixels
            tree_pixels = 0
            total_pixels = np.sum(segmentation_mask > 0)  # Total number of non-zero pixels
            for segment, label in zip(results['segments_info'], labels):
                segment_label = model.config.id2label[segment['label_id']]
                if 'tree' in segment_label.lower():
                    mask = (segmentation_mask == segment['id']).astype(np.uint8)
                    segment_pixels = cv2.countNonZero(mask)
                    tree_pixels += segment_pixels

            gvi = tree_pixels / total_pixels
            total_gvi += gvi
            num_images += 1

    # Calculate average GVI
    avg_gvi = total_gvi / num_images if num_images > 0 else 0
    avg_gvi_label.config(text=f"Average GVI across all images: {avg_gvi:.2f}")

def insert_colored_text(text, color):
    result_text.tag_config(color, foreground=color)
    result_text.insert(tk.END, text, color)

def plot_destination_locations(latitude, longitude, destination_locations):
    # Calculate the bounding box of all markers (origin and destinations)
    all_locations = [(latitude, longitude)] + destination_locations
    min_lat = min(lat for lat, lng in all_locations)
    max_lat = max(lat for lat, lng in all_locations)
    min_lng = min(lng for lat, lng in all_locations)
    max_lng = max(lng for lat, lng in all_locations)

    # Calculate the center and zoom level of the map
    center_lat = (min_lat + max_lat) / 2
    center_lng = (min_lng + max_lng) / 2
    zoom_level = calculate_zoom_level(min_lat, max_lat, min_lng, max_lng)

    # Create a map plot centered at the calculated center with the calculated zoom level
    gmap = gmplot.GoogleMapPlotter(center_lat, center_lng, zoom_level)

    # Add origin marker
    origin_color = color_tags["origin"]["marker_color"]
    gmap.marker(latitude, longitude, origin_color)

    # Add destination markers with corresponding colors
    for i, (lat, lng) in enumerate(destination_locations):
        destination_color = color_tags[f"destination{i+1}"]["marker_color"]
        gmap.marker(lat, lng, destination_color)

    # Draw the map to an HTML file
    gmap.draw("static_map.html")

    # Open the generated HTML file in the default web browser
    import webbrowser
    webbrowser.open("static_map.html")

import math
# Ensure zoom level is non-negative
# Ensure zoom level is non-negative
# Ensure zoom level is non-negative


def calculate_zoom_level(min_lat, max_lat, min_lng, max_lng):
    # Function to calculate the zoom level based on the bounding box of coordinates
    WORLD_DIM = {'height': 120, 'width': 120}  # Adjust these values as per your requirement
    ZOOM_MAX = 15  # Maximum zoom level

    lat_diff = max_lat - min_lat
    lng_diff = max_lng - min_lng

    lat_zoom = math.ceil(math.log2(WORLD_DIM['height'] * 360 / lat_diff))
    lng_zoom = math.ceil(math.log2(WORLD_DIM['width'] * 360 / lng_diff))

    zoom_level = min(lat_zoom, lng_zoom, ZOOM_MAX)
    return max(0, zoom_level)  # Ensure zoom level is non-negative




def calculate_distance_matrix():
    global result_text  # Accessing the global result_text widget

    latitude = float(latitude_entry.get())
    longitude = float(longitude_entry.get())
    keyword = keyword_entry.get()

    # Define the location using the obtained latitude and longitude
    location = (latitude, longitude)

    # Area within 1000 m of the specified location
    radius = 1000  # Radius in meters

    # Retrieve nearby places around the specified location based on the keyword
    desirable_places = client.places_nearby(location=location, radius=radius, keyword=keyword)

    # List to store the destinations for the distance matrix
    destination_locations = []

    # Initialize count of results with duration less than 7 minutes
    count_less_than_7_mins = 0

    # Iterate over each place and geocode its vicinity address
    for i, place in enumerate(desirable_places['results']):
        vicinity_address = place.get('vicinity')

        # Geocode the vicinity address
        geocoding_result = client.geocode(address=vicinity_address)

        # Extract the geographic coordinates (latitude and longitude) from the geocoding result
        if geocoding_result:
            vicinity_latitude = geocoding_result[0]['geometry']['location']['lat']
            vicinity_longitude = geocoding_result[0]['geometry']['location']['lng']

            # Append the geocoded coordinates to the destination_locations list
            destination_locations.append((vicinity_latitude, vicinity_longitude))

            # Calculate the distance and duration
            distance = "N/A"
            duration = "N/A"
            duration_value = float('inf')  # Default to infinity
            if destination_locations:
                origin = location
                destination = (vicinity_latitude, vicinity_longitude)
                distance_matrix_result = client.distance_matrix(
                    origins=[origin],
                    destinations=[destination],
                    mode="driving",
                    language="en",
                    units="metric",
                    departure_time="now",
                    traffic_model="best_guess",
                    region="in"
                )
                if 'rows' in distance_matrix_result:
                    elements = distance_matrix_result['rows'][0]['elements']
                    if elements and 'distance' in elements[0] and 'duration' in elements[0]:
                        distance = elements[0]['distance']['text']
                        duration = elements[0]['duration']['text']
                        if 'value' in elements[0]['duration']:
                            duration_value = elements[0]['duration']['value']
                            if duration_value < 420:  # Less than 7 minutes (7*60 = 420 seconds)
                                count_less_than_7_mins += 1

            # Insert text with appropriate color tag
            insert_colored_text(f"Name: {place.get('name')}\n", color_tags[f"destination{i+1}"]["text_color"])
            insert_colored_text(f"Location: Latitude: {vicinity_latitude}, Longitude: {vicinity_longitude}\n", color_tags[f"destination{i+1}"]["text_color"])
            insert_colored_text(f"Rating: {place.get('rating')}\n", color_tags[f"destination{i+1}"]["text_color"])
            insert_colored_text(f"Distance: {distance}, Duration: {duration}\n", color_tags[f"destination{i+1}"]["text_color"])
            insert_colored_text("----------\n", color_tags[f"destination{i+1}"]["text_color"])

    # Display total count of results with duration less than 7 minutes
    insert_colored_text(f"Total Results with Duration Less than 7 Minutes: {count_less_than_7_mins}\n", "black")

    # Plot destination locations on the map with corresponding colors
    plot_destination_locations(latitude, longitude, destination_locations)



   
# Create Tkinter window



# Function to show average GVI

# Create main window
# Create Tkinter window
window = tk.Tk()
window.title("Street View and Air Quality Data")

# Set full screen
window.attributes('-fullscreen', True)

# Set background image (Note: You need to have bgimage.jpg in the specified location)
background_image = Image.open(r"C:\Users\sthdh\Downloads\bgimage.jpg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = tk.Label(window, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


# Configure font style
font_style = ("Helvetica", 12)

# Latitude label and input
latitude_label = tk.Label(window, text="Latitude:", font=font_style, fg="dark green")
latitude_label.pack(anchor="center", pady=5)
latitude_entry = tk.Entry(window, font=font_style)
latitude_entry.pack(anchor="center", pady=5)

# Longitude label and input
longitude_label = tk.Label(window, text="Longitude:", font=font_style, fg="dark green")
longitude_label.pack(anchor="center", pady=5)
longitude_entry = tk.Entry(window, font=font_style)
longitude_entry.pack(anchor="center", pady=5)

# File name prefix label and input
file_name_label = tk.Label(window, text="File name prefix:", font=font_style, fg="dark green")
file_name_label.pack(anchor="center", pady=5)
file_name_entry = tk.Entry(window, font=font_style)
file_name_entry.pack(anchor="center", pady=5)

# Save location label and input
save_location_label = tk.Label(window, text="Save location:", font=font_style, fg="dark green")
save_location_label.pack(anchor="center", pady=5)
save_location_entry = tk.Entry(window, font=font_style)
save_location_entry.pack(anchor="center", pady=5)

# Keyword label and input
keyword_label = tk.Label(window, text="Keyword:", font=font_style, fg="dark green")
keyword_label.pack(anchor="center", pady=5)
keyword_entry = tk.Entry(window, font=font_style)
keyword_entry.pack(anchor="center", pady=5)

# Button to trigger street view action
street_view_button = tk.Button(window, text="Download Street View", command=on_street_view_click, font=font_style, bg="dark green", fg="white", padx=10, pady=5)
street_view_button.pack(anchor="center", pady=5)

# Button to trigger air quality action
air_quality_button = tk.Button(window, text="Fetch Air Quality", command=on_air_quality_click, font=font_style, bg="dark green", fg="white", padx=10, pady=5)
air_quality_button.pack(anchor="center", pady=5)

# Button to calculate average GVI
calculate_button = tk.Button(window, text="Calculate Distance Matrix", command=calculate_distance_matrix, font=font_style, bg="dark green", fg="white", padx=10, pady=5)
calculate_button.pack(anchor="center", pady=5)

# Label to display average GVI
avg_gvi_label = tk.Label(window, text=" Wait it's calculating ", font=font_style, bg="dark blue", fg="white", padx=10, pady=5)
avg_gvi_label.pack(pady=10)

# Text widget to display the results
result_text = tk.Text(window, height=20, width=50)
result_text.pack(anchor="center", pady=5)

# Start the Tkinter event loop
window.mainloop()

# Run the Tkinter event loop
window.mainloop()
