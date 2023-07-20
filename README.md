# Wound Sernor Registration




## Table of Contents

- [About](#about)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## About

In this code, we are working with a sensor that changes its color based on the amount of wound fluid it absorbs and the sample concentration. The main objective is to explore any correlation between the color change and concentration.

To achieve this, we perform image registration between the sensor images taken before and after exposure to the wound. Image registration is a technique used to align images with each other, allowing us to compare corresponding regions accurately. However, we faced challenges with standard image registration methods, especially when the sensor images were taken from the left or right side. Due to this limitation, we decided to adopt an alternative approach.

Instead of traditional image registration, we employ color filtering and utilize the circular geometry of the sensors to detect and analyze them. The process involves extracting regions of interest from the images, representing the sensors, and analyzing color changes within these regions.

Here's a brief overview of the key steps in our approach:

Image Preprocessing: We perform color filtering to isolate the sensors and extract regions of interest for further analysis.

Sensor Detection: The extracted regions are processed to identify circular sensor structures based on their geometries.

Color Change Analysis: We calculate color changes in the sensor regions between images taken before and after wound exposure.

Concentration Relation: Finally, we explore any relationship between the color change and the sample concentration.

By adopting this approach, we aim to gain insights into the color change behavior of the sensors, especially concerning different sample concentrations. This analysis can potentially lead to a better understanding of the sensor's performance and its correlation with varying concentrations, which may have implications for wound monitoring and healthcare applications.

## Installation

To install the required packages, run the following command:

`pip install -r requirements.txt`

## Usage

`python3 main.py {path to dry image} {path to exposed image} {path to store the result}`

### Jupyter Notebook: `align_images.ipynb`

A scratch notebook which contains the code for different experiments that were performed.

### Scripts

- `main.py`: Main function.
- `color_utils.py`: 
    - `extractColoredBoundingBox`: This function extracts the bounding box of a colored object in an image based on color filtering.
    - `getFinalPts`: It calculates the centroids and contours of colored objects detected in the image.
    - `getHeatMapColorDifference`: This function generates the heatmap difference between two images, highlighting the areas with color changes
- `image_utils.py`:
    - `extract_yellowcircles`: This function extracts yellow circles from the input images.
    - `changeFormat`: It converts the color format of the images, such as BGR to RGB.
- `plots.py`: Contains various functions to generate different types of plots using the data extracted from the input images
- `utils.py`: The script provides a comprehensive set of functions to analyze the color changes in sensor images and establish a correlation between the color change and concentration of the wound fluid.
    - `calculateSimilarity`: Calculates the similarity between two images using color difference and heatmap generation. The function extracts regions of interest, computes color change metrics, and returns the cropped images with overlaid heatmaps.
    - `generateResultsSingleImage`: Performs image processing and analysis for a single pair of base and sample images. It displays and saves the processed images, along with metrics for color change in each circle.
    - `generateResultsBulk`: Executes bulk image processing and analysis for multiple sample images. It iterates through the sample folders, processes the images, and generates results in the specified results folder. It also saves the results data in a CSV file.

### Images

- `sensor.jpg`: Sample view of the sensor.
- `results/sample_results.png`: Sample result image.

## Results



### Plots

- `results/intensity_change_box.png`: intensity change box plot for different concentration.
- `results/pixels_changed_box.png`: Percentage of pixels changed box plot for different concentration.

### Data

- `resultsData.csv`: Save results data after processing bulk image processing and analysis for multiple sample images.


