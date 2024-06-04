import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2
import math
import os
import re
from color_utils import extractColoredBoundingBox, getFinalPts, getHeatMapColorDifference
from image_utils import extract_yellowcircles, changeFormat
import matplotlib.gridspec as gridspec
import pandas as pd

def extractRegions(imFilename):
    paddingFactor = 0.4
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    ima, cropped_im,_,_, yellowArea,yellow_objs = extract_yellowcircles(im, paddingFactor, True)
    if (len(yellow_objs) == 0):
        print('1st phase of yellow objects scanning did not detect sensor')
        return None, cropped_im
    # Display the annotated image
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cropped_im)
    # plt.title('Cropped')
    # plt.show()
    # plt.figure(figsize=(10, 8))
    # plt.imshow(changeFormat(whitebbIm))
    # plt.title('white')
    # plt.show()

    # im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    # print(cropped_im)

    ima, cropped_im ,_, tmp_img, yellowArea, yellow_objs = extract_yellowcircles(cropped_im, paddingFactor, False)

    if (len(yellow_objs) == 0):
        print('2nd phase of yellow objects scanning did not detect sensor')
        return None, cropped_im


    for (x,y,bb) in yellow_objs:
        rectX, rectY, rectWidth, rectHeight = cv2.boundingRect(bb[0])
        # cv2.rectangle(cropped_im, (int(rectX), int(rectY)),
        #               (int(rectX + rectWidth), int(rectY + rectHeight)), (255,255,0), 2)

    # Calculate the centroid of the four centroids
    # centroid_x = np.mean([x[0] for x in yellow_objs])
    # centroid_y = np.mean([x[1] for x in yellow_objs])

    # # Calculate the angle of rotation
    # angle = np.arctan2(yellow_objs[1][1] - yellow_objs[0][1], yellow_objs[1][0] - yellow_objs[0][0])
    # angle_degrees = np.degrees(angle)

    # # Rotate each structure around the centroid
    # rotated_yellow_objs = []
    # for (x, y, bb) in yellow_objs:
    #     # Translate the coordinates to the origin (centroid)
    #     translated_x = x - centroid_x
    #     translated_y = y - centroid_y

    #     # Perform the rotation
    #     rotated_x = translated_x * np.cos(angle) - translated_y * np.sin(angle)
    #     rotated_y = translated_x * np.sin(angle) + translated_y * np.cos(angle)

    #     # Translate the coordinates back to the original position
    #     rotated_x += centroid_x
    #     rotated_y += centroid_y

    #     # Append the rotated coordinates and contour to the list
    #     rotated_yellow_objs.append((rotated_x, rotated_y, bb))

    # # Draw the rotated rectangles
    # for (x, y, bb) in rotated_yellow_objs:
    #     rectX, rectY, rectWidth, rectHeight = cv2.boundingRect(bb[0])
    #     cv2.rectangle(cropped_im, (int(rectX), int(rectY)), (int(rectX + rectWidth), int(rectY + rectHeight)), (255, 255, 0), 2)

    #     cv2.circle(cropped_im ,(x,y), 2, (255, 0, 0), 2)

    # for (x,y) in black_objs:
    #     cv2.circle(cropped_im ,(x,y), 2, (0, 255, 0), 2)

    # for (x,y) in white_objs:
    #     cv2.circle(cropped_im ,(x,y), 2, (0, 0, 255), 2)

    # plt.figure()
    # plt.imshow(changeFormat(ima))
    # plt.figure(figsize=(10,8))
    # plt.imshow(changeFormat(cropped_im))
    # plt.title('Yellow')
    # plt.show()
    # try:
    # print("White Objects: ", len(white_objs), 'Black Objects: ' , len(black_objs), 'Yellow Objects: ', len(yellow_objs))
    # Removing this part because the order of the yellow regions is not needed.
    # final_pts = getFinalPts(black_objs, white_objs, yellow_objs)
    final_pts = yellow_objs
    return final_pts, cropped_im

def compute_color_similarity(croppedBaseImage, contour1, croppedImage,contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    roi1 = croppedBaseImage[y1:y1+h1, x1:x1+w1]
    roi2 = croppedImage[y2:y2+h2, x2:x2+w2]

    hist1 = cv2.calcHist([roi1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([roi2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return similarity

def shrinkBoundary(contour, percent_shrink):
    # Calculate centroid of the contour
    moments = cv2.moments(contour)
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    # Determine the dimensions of the bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate the shrinkage offsets based on the bounding box dimensions
    shrinkage_offset_x = int(w * percent_shrink / 200)
    shrinkage_offset_y = int(h * percent_shrink / 200)

    # Create mask image
    mask = np.zeros_like(contour)

    # Iterate over contour points and update the mask image
    for i in range(len(contour)):
        point = contour[i][0]
        x, y = point[0], point[1]
        dx = centroid_x - x
        dy = centroid_y - y
        dist = np.sqrt(dx*dx + dy*dy)
        if dist > 0:
            new_x = int(x + (dx / dist) * shrinkage_offset_x)
            new_y = int(y + (dy / dist) * shrinkage_offset_y)
            mask[i] = [new_x, new_y]
        else:
            mask[i] = [x, y]

    # Extract the shrunken blob using the updated mask
    shrunken_blob = mask.reshape((-1, 1, 2)).astype(np.int32)

    return shrunken_blob

# Calculate the angle between two lines
def calculateAngle(p1, p2, p3, p4):
    angle1 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = np.arctan2(p4[1] - p3[1], p4[0] - p3[0])
    angle_degrees = np.degrees(angle2 - angle1)
    return angle_degrees



def calculateSimilarity(baseImage, image, intensityThresh = 0, pixelsChangedThresh = 30):
    print('Working on baseImage..')
    baseImageRegions, croppedBaseImage = extractRegions(baseImage)
    print('Working on image..')
    imageRegions, croppedImage = extractRegions(image)
    if(baseImageRegions is None):
        print('All regions not detected in base image.')
        return None,None, None, None, None

    if(imageRegions is None ):
        print('All regions not detected in image.')
        return None,None, None, None, None

    baseImageHeatMaps = []
    imageHeatMaps = []
    main_background = np.zeros_like(croppedBaseImage)
    metrics = []
    
    # Calculate the angle to rotate the line formed by points ((imageRegions[0][0],imageRegions[0][1]) and (imageRegions[1][0], imageRegions[1][1])
    # to match the line formed by points ((imageRegions[2][0],imageRegions[2][1]) and (imageRegions[3][0], imageRegions[3][1])
    # plt.figure()
    # plt.imshow(imutils.rotate(croppedImage, -angle_degrees))
    print(len(imageRegions), len(baseImageRegions))
    if not ((len(baseImageRegions) == len(imageRegions)) and len(baseImageRegions) == 4):
        print("Missing regions, Base Image Regions:", len(baseImageRegions), ' Image Regions: ', len(imageRegions))
        return None, None, None, None, None

    for i in range(len(baseImageRegions)):

        if(imageRegions is None):
            print("Missing image regions", i)
            return None, None, None, None, None
        
        if(baseImageRegions is None):
            print('Missing baseImage regions', i)
            return None,None, None, None, None

        angle_degrees = calculateAngle(imageRegions[0], imageRegions[1], baseImageRegions[0], baseImageRegions[1])
        baseContour = baseImageRegions[i][2][0]
        baseContour = shrinkBoundary(baseContour, 10)
        imageContour = imageRegions[i][2][0]
        
        x1, y1, w1, h1 = cv2.boundingRect(baseContour)
        border_percentage = -0.05
        border_width = int(border_percentage * w1)
        border_height = int(border_percentage * h1) 
        x1 += border_width
        y1 += border_height
        w1 -= 2 * border_width
        h1 -= 2 * border_height

        # Extract the bounding box of contour2 in image2
        x2, y2, w2, h2 = cv2.boundingRect(imageContour)
        border2_percentage = 0.1
        border2_width = int(border2_percentage * w2)
        border2_height = int(border2_percentage * h2)
        x2 += border2_width
        y2 += border2_height
        w2 -= 2 * border2_width
        h2 -= 2 * border2_height
        roi2 = croppedImage[y2:y2+h2, x2:x2+w2]
    
        # Resize roi2 to match the size of contour1 in image1
        rotated_roi2 = imutils.rotate(roi2, -angle_degrees)
        center_x = rotated_roi2.shape[1] // 2
        center_y = rotated_roi2.shape[0] // 2

        # Calculate the region of interest coordinates
        x_roi = center_x - (w2 // 2)
        y_roi = center_y - (h2 // 2)
        x_roi_end = x_roi + w2
        y_roi_end = y_roi + h2

        # Extract the region of interest from rotated_resized_roi2
        fitted_roi2 = rotated_roi2[y_roi:y_roi_end, x_roi:x_roi_end]
        rotated_resized_roi2 = cv2.resize(fitted_roi2, (w1, h1))
        
        # Generate the heatmap difference on the resized ROI of image1
        heatmap_diff = getHeatMapColorDifference(croppedBaseImage[y1:y1+h1, x1:x1+w1], rotated_resized_roi2, 'min-max')

        # Create a mask for contour1
        contour_mask = np.zeros_like(croppedBaseImage)
        cv2.drawContours(contour_mask, [baseContour], 0, (255, 255, 255), -1)
     
        # background_tmp = np.zeros_like(croppedBaseImage)
        background = np.zeros_like(croppedBaseImage)
        image_diff = cv2.absdiff(croppedBaseImage[y1:y1+h1, x1:x1+w1], rotated_resized_roi2)
        # Overlay the heatmap difference on the background image at the position of contour1
        # background_tmp[y1:y1+h1, x1:x1+w1] = image_diff
        background[y1:y1+h1, x1:x1+w1] = heatmap_diff

        # Apply the contour mask to the background image
        # masked_background_tmp = cv2.bitwise_and(background, contour_mask)
        masked_background = cv2.bitwise_and(background, contour_mask)

        # Calculate intensity of change within the contour region
        intensity_change = np.mean(masked_background[contour_mask > intensityThresh])


        # Calculate the percentage of changed pixels within the contour region
        total_pixels = np.count_nonzero(contour_mask)
        changed_pixels = np.count_nonzero(masked_background)
        percentage_changed = (changed_pixels / total_pixels) * 100
        # Apply the contour mask to the background image
        background = cv2.bitwise_and(masked_background, contour_mask)
        main_background = cv2.add(main_background, background)
        contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2GRAY) # Convert to grayscale
        # contour_mask = cv2.threshold(contour_mask, 1, 255, cv2.THRESH_BINARY)[1] # Apply binary threshold

        # Compute histogram for each contour in main_background
    #     hist = cv2.calcHist([main_background], [0], contour_mask, [256], [50, 250])
    #     hist = hist.flatten()

    #    # Normalize histogram values to sum up to 1
    #     hist /= hist.sum()

    #     # Plot the histogram as a bar graph
    #     plt.figure()
    #     plt.bar(range(len(hist)), hist)
    #     plt.title(f"Histogram - Contour {i+1}")
    #     plt.xlabel("Intensity")
    #     plt.ylabel("Frequency")


        metrics.append((percentage_changed, intensity_change))
        # metrics.append({'changedPer': percentage_changed, "intensityChange": intensity_change})
        # Add the background image with the heatmap to image1

    

        # similarity.append(compute_color_similarity(croppedBaseImage, baseContour, croppedImage, imageContour))
        # print("Color similarity between contour", i+1, "in dry image and ", folderName ," image:", similarity)
        
    
    # # Display the heatmaps
    # plt.figure(figsize=(12, 6))
    # for i in range(len(baseImageHeatMaps)):
    #     plt.subplot(2, len(baseImageHeatMaps), i+1)
    #     plt.imshow(cv2.cvtColor(baseImageHeatMaps[i], cv2.COLOR_BGR2RGB))
    #     plt.title("Base Image Heatmap {}".format(i+1))
    #     plt.axis("off")

    #     plt.subplot(2, len(imageHeatMaps), len(baseImageHeatMaps)+i+1)
    #     plt.imshow(cv2.cvtColor(imageHeatMaps[i], cv2.COLOR_BGR2RGB))
    #     plt.title("Image Heatmap {}".format(i+1))
    #     plt.axis("off")
    result = cv2.addWeighted(croppedBaseImage, 0.6, main_background, 0.4, 0)
    return croppedBaseImage, croppedImage, main_background, result, metrics

def generateResultsBulk(input_folder = "stand images", results_folder = "results"):
    entries = []

    # Specify the base folder and the results folder
    input_folder = "/Users/apparilalith/Desktop/asu/mayolab/woundsensor/input_images/iphone"
    results_folder = "results"

    success_cnt = 0
    failure_cnt = 0
    # Create the results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    sample_dirs = [f for f in os.listdir(input_folder) if not (f.startswith('.') or f.endswith('.db'))]
    # Iterate over the sample folders
    for sample_name in sample_dirs:
        sample_folder = os.path.join(input_folder, sample_name)
        base_folder = os.path.join(input_folder, sample_name)
        dirs = [f for f in os.listdir(sample_folder) if not (f.startswith('.') or f.endswith('.db'))]
        for sample_type in dirs:
            image_folder = os.path.join(sample_folder, sample_type)
            base_dry_folder = os.path.join(base_folder, 'Dry')
            for number in range(1, 11):
                base_image_path = None
                image_path = None

                # Iterate over the files in the image folder
                for filename in os.listdir(image_folder):
                    match = re.search(r"S(\d+).*\.jpg", filename)
                    if filename.endswith(".jpg") and (match and int(match.group(1)) == number):
                        image_path = os.path.join(image_folder, filename)

                # Iterate over the files in the base dry folder
                for filename in os.listdir(base_dry_folder):
                    match = re.search(r"S(\d+).*\.jpg", filename)
                    if filename.endswith(".jpg") and (match and int(match.group(1)) == number):
                        base_image_path = os.path.join(base_dry_folder, filename)
                # Create the result folder path based on the image_path
                if image_path is None or base_image_path is None:
                    print("No file found : ", str(sample_type)+str(number))
                    continue
                relative_folder_path = os.path.dirname(image_path)
                relative_folder_path = os.path.relpath(relative_folder_path, input_folder)
                result_folder_path = os.path.join(results_folder, relative_folder_path)

                croppedBaseImage, croppedImage, main_background, result, metrics = calculateSimilarity(base_image_path, image_path)
                if(metrics == None):
                    print(image_path)
                    failure_cnt+=1
                    continue
                success_cnt+=1
                # metrics has four entries for each circle, each circle has 2 measures number of pixels changed, intensity change
                print(sample_name, sample_type,metrics)
                # Add the entry to the DataFrame
                entry = {
                    'Concentration': sample_name,
                    'Sample Type': sample_type,
                    'Sample Number': number,
                    'Circle 1 Pixels Changed': metrics[0][0],
                    'Circle 1 Intensity Change': metrics[0][1],
                    'Circle 2 Pixels Changed': metrics[1][0],
                    'Circle 2 Intensity Change': metrics[1][1],
                    'Circle 3 Pixels Changed': metrics[2][0],
                    'Circle 3 Intensity Change': metrics[2][1],
                    'Circle 4 Pixels Changed': metrics[3][0],
                    'Circle 4 Intensity Change': metrics[3][1]
                }
                entries.append(entry)


                # Create the result folder if it doesn't exist
                os.makedirs(result_folder_path, exist_ok=True)
                # Construct the result filepath
                result_filename = f"result_{sample_name}_{sample_type}_{number}.png"
                result_filepath = os.path.join(result_folder_path, result_filename) 

                # Display and save the figures
                plt.figure(figsize=(16, 10))
                plt.subplot(1, 4, 1)
                # plt.imshow(cv2.cvtColor(croppedBaseImage, cv2.COLOR_BGR2RGB))
                plt.imshow(cv2.cvtColor(cv2.imread(base_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                plt.title("Base Image")
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                # plt.imshow(cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))
                plt.title("Image")
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(cv2.cvtColor(main_background, cv2.COLOR_BGR2RGB))
                plt.title("Heatmap Difference")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.imshow(cv2.cvtColor(main_background, cv2.COLOR_BGR2RGB), cmap='jet', interpolation='none', alpha=0.35)
                plt.title("Overlay")
                plt.axis("off")
                cbar = plt.colorbar(aspect=8)  # Adjust the aspect ratio here
                cbar.ax.tick_params(labelsize=8)

                plt.tight_layout()
                plt.savefig(result_filepath)
                plt.close()

    # Additional operations if needed
    df = pd.DataFrame(entries)
    df.to_csv('resultsData.csv', index=False)
    print("Failed cases: ", failure_cnt, 'Succesfull cases:', success_cnt)
    return df

def generateResultsSingleImage(base_image_path, image_path, result_filepath):
    print(base_image_path, image_path)
    # Create the result folder path based on the image_path
    if image_path is None or base_image_path is None:
        print("No file found  ")
        return 
    # Calculate similarity and metrics
    croppedBaseImage, croppedImage, main_background, result, metrics = calculateSimilarity(base_image_path, image_path)

    if croppedBaseImage is None or croppedImage is None:
        print('Error while processing the images')
        return
    # Get the dimensions of the images
    base_image_height, base_image_width, _ = croppedBaseImage.shape
    image_height, image_width, _ = croppedImage.shape
    
    # Create a custom grid for the subplots
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.5, 1])

    # Display and save the figures
    plt.figure(figsize=(16, 4))

    # Subplot 1: Base Image
    plt.subplot(gs[0])
    plt.imshow(cv2.cvtColor(croppedBaseImage, cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(cv2.imread(base_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    plt.title("Base Image")
    plt.axis("off")

    # Subplot 2: Image
    plt.subplot(gs[1])
    plt.imshow(cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")

    # Subplot 3: Metrics for Circles
    ax1 = plt.subplot(gs[2])
    # Extracting the circle numbers and corresponding values
    circle_numbers = list(range(1, 5))  # Assuming there are 4 circles (adjust accordingly)
    pixels_changed = [metrics[i][0] for i in range(4)]
    intensity_change = [metrics[i][1] for i in range(4)]

    # Plotting pixels changed on the primary y-axis
    ax1.plot(circle_numbers, pixels_changed, label='Pixels Changed', marker='o', color='tab:blue')
    ax1.set_xlabel('Circle Number')
    ax1.set_ylabel('Pixels Changed', color='tab:blue')

    # Creating a secondary y-axis for intensity change
    ax2 = ax1.twinx()
    ax2.plot(circle_numbers, intensity_change, label='Intensity Change', marker='o', color='tab:red')
    ax2.set_ylabel('Intensity Change', color='tab:red')

    # Adding title
    plt.title('Metrics for Circles')

    # Set x-axis tick positions and labels to display as "Circle 1", "Circle 2", etc.
    ax1.set_xticks(circle_numbers)
    ax1.set_xticklabels([f'Circle {i}' for i in circle_numbers])

    # Adding legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # plt.show()

    # Subplot 4: Overlay with Colorbar
    plt.subplot(gs[3])
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.imshow(cv2.cvtColor(main_background, cv2.COLOR_BGR2RGB), cmap='jet', interpolation='none', alpha=0.35)
    plt.title("Overlay")
    plt.axis("off")
    cbar = plt.colorbar(aspect=40)  # Adjust the aspect ratio for the colorbar here
    cbar.ax.tick_params(labelsize=8)

    print(os.path.basename(image_path).split('.')[0])
    # Adjust the layout and save the plot
    plt.tight_layout()
    result_file_path = os.path.join(result_filepath, 'results_' + str(os.path.basename(image_path).split('.')[0]) + '.png')
    print(result_file_path)
    plt.savefig(result_file_path)
    plt.close()

    # plt.close()
    return result_file_path, metrics





