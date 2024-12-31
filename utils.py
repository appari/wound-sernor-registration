import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2
import math
import os
import re
from color_utils import extractColoredBoundingBox, getFinalPts, getHeatMapColorDifference
from image_utils import extract_yellowcircles, changeFormat, extracting_sensor_region
import matplotlib.gridspec as gridspec
import pandas as pd


def extract_background_color_value(image):
    x, y, w, h = extracting_sensor_region(image)
    border_percentage = 0.2
    border_width = int(border_percentage * w)
    border_height = int(border_percentage * h)

    x2 = max(0, x - border_width)
    y2 = max(0, y - border_height)
    x3 = min(image.shape[1], x + w + border_width)
    y3 = min(image.shape[0], y + h + border_height)

    strip = image[y2:y3, x2:x3].copy()

    sensor_x_start = x - x2
    sensor_y_start = y - y2
    strip[sensor_y_start:sensor_y_start + h,
          sensor_x_start:sensor_x_start + w] = 0

    background_color_value = np.mean(
        np.dot(strip[..., :3], [0.2989, 0.587, 0.114]))

    return background_color_value


def extractRegions(imFilename):
    paddingFactor = 0.4
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    ima, cropped_im, _, _, yellowArea, yellow_objs = extract_yellowcircles(im, paddingFactor, True)
   

    # background_color_value = extract_background_color_value(im)

    if (len(yellow_objs) == 0):
        print('1st phase of yellow objects scanning did not detect sensor')
        return None, cropped_im, None
    # Display the annotated image

    # im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    # print(cropped_im)

    ima, cropped_im ,_, tmp_img, yellowArea, yellow_objs = extract_yellowcircles(cropped_im, paddingFactor, False)

    for (x, y, bb) in yellow_objs:
        rectX, rectY, rectWidth, rectHeight = cv2.boundingRect(bb[0])
        # cv2.rectangle(cropped_im, (int(rectX), int(rectY)),
        #               (int(rectX + rectWidth), int(rectY + rectHeight)), (255,255,0), 2)

    final_pts = yellow_objs
    return final_pts, cropped_im, None


def compute_color_similarity(croppedBaseImage, contour1, croppedImage, contour2):
    x1, y1, w1, h1 = cv2.boundingRect(contour1)
    x2, y2, w2, h2 = cv2.boundingRect(contour2)
    roi1 = croppedBaseImage[y1:y1+h1, x1:x1+w1]
    roi2 = croppedImage[y2:y2+h2, x2:x2+w2]

    hist1 = cv2.calcHist([roi1], [0, 1, 2], None, [
                         8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([roi2], [0, 1, 2], None, [
                         8, 8, 8], [0, 256, 0, 256, 0, 256])

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


def is_outlier(value, lower, upper):
    """Check if value is an outlier
    """
    return value <= lower or value >= upper


def find_outlier_and_avg(array):
    # Calculate the differences between elements of each tuple
    differences = [abs(t[0] - t[1]) for t in array]

    # Calculate quartiles for the differences
    q1 = sorted(differences)[0]
    q3 = sorted(differences)[-1]
    q2 = (q1 + q3) / 2

    # Calculate outlier bounds for the differences
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Find the outlier
    outlier = [value for value in differences if is_outlier(
        value, lower, upper)]
    outlier = outlier[0] if outlier else None

    # Calculate the average of the first element of tuples in the other set
    other_set = [t for t in array if abs(t[0] - t[1]) != outlier]
    avg_other_set = sum(t[1] for t in other_set) / \
        len(other_set) if other_set else None

    return np.array([avg_other_set])
    # print('Other set', other_set)
    # return np.array(other_set)


import cv2
import numpy as np

def calculateSimilarity(baseImage, image, intensityThresh=0, pixelsChangedThresh=30, use_reinhard=False):
    print('Working on baseImage..')
    baseImageRegions, croppedBaseImage, basebackground_value = extractRegions(baseImage)
    print('Working on image..')
    imageRegions, croppedImage, background_value = extractRegions(image)

    if baseImageRegions is None or len(baseImageRegions) == 0:
        print('No valid regions detected in base image.')
        return None, None, None, None, None

    if imageRegions is None or len(imageRegions) == 0:
        print('No valid regions detected in target image.')
        return None, None, None, None, None

    main_background = np.zeros_like(croppedBaseImage)
    metrics = []

    def reinhard_color_transfer(source_mean, source_std, target):
        if len(target.shape) == 2:
            print("Reinhard conversion not applicable to grayscale images.")
            return target
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        t_mean, t_std = cv2.meanStdDev(target_lab)
        t_mean = t_mean.reshape((1, 1, 3))
        t_std = t_std.reshape((1, 1, 3))
        source_mean = source_mean.reshape((1, 1, 3))
        source_std = source_std.reshape((1, 1, 3))
        normalized_lab = (target_lab - t_mean) / t_std * source_std + source_mean
        normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    def crop_assay(image, contour, dims):
        x, y, w, h = dims
        contour_mask = np.zeros_like(image)
        cv2.drawContours(contour_mask, [contour], 0, (255, 255, 255), -1)
        background = np.zeros_like(image)
        background[y:y + h, x:x + w] = image[y:y + h, x:x + w]
        return cv2.bitwise_and(background, contour_mask)

    all_base_pixels = []
    for region in baseImageRegions:
        baseContour = region[2][0]
        x, y, w, h = cv2.boundingRect(baseContour)
        base_assay = crop_assay(croppedBaseImage, baseContour, (x, y, w, h))
        lab_assay = cv2.cvtColor(base_assay, cv2.COLOR_BGR2LAB).astype(np.float32)
        valid_pixels = lab_assay[lab_assay[:, :, 0] > 0]
        all_base_pixels.append(valid_pixels)

    all_base_pixels = np.vstack(all_base_pixels)
    source_mean, source_std = np.mean(all_base_pixels, axis=0), np.std(all_base_pixels, axis=0)
    print(f"Global Base Image Mean: {source_mean}, Std: {source_std}")

    # Initialize dictionary to store metrics
    metrics = {
        "base_image": [],
        "target_image": []
    }

    # Process target image regions
    for i in range(min(len(baseImageRegions), len(imageRegions))):
        # Target Image Assay
        imageContour = imageRegions[i][2][0]
        x, y, w, h = cv2.boundingRect(imageContour)
        image_assay = crop_assay(croppedImage, imageContour, (x, y, w, h))

        if use_reinhard:
            calibrated_assay = reinhard_color_transfer(source_mean, source_std, image_assay)
        else:
            calibrated_assay = image_assay

        if len(calibrated_assay.shape) == 3 and calibrated_assay.shape[2] == 3:
            # Calculate channel-wise mean for non-background pixels
            channel_means = [np.mean(calibrated_assay[:, :, c][calibrated_assay[:, :, c] > 0]) for c in range(3)]
        elif len(calibrated_assay.shape) == 2:
            channel_means = [np.mean(calibrated_assay[calibrated_assay > 0])] * 3  # Treat grayscale as uniform across channels
        else:
            raise ValueError("Unsupported channel format for calibrated assay.")

        metrics["target_image"].append({
            "region_id": i + 1,
            "channel_means": channel_means
        })
        print(f"Target Image Assay {i + 1} - Channel-Wise Mean Intensities: {channel_means}")

    # Process base image regions
    for i, region in enumerate(baseImageRegions):
        baseContour = region[2][0]
        x, y, w, h = cv2.boundingRect(baseContour)
        base_assay = crop_assay(croppedBaseImage, baseContour, (x, y, w, h))

        if len(base_assay.shape) == 3 and base_assay.shape[2] == 3:
            base_channel_means = [np.mean(base_assay[:, :, c][base_assay[:, :, c] > 0]) for c in range(3)]
        elif len(base_assay.shape) == 2:
            base_channel_means = [np.mean(base_assay[base_assay > 0])] * 3
        else:
            raise ValueError("Unsupported channel format for base assay.")

        metrics["base_image"].append({
            "region_id": i + 1,
            "channel_means": base_channel_means
        })
        print(f"Base Image Assay {i + 1} - Channel-Wise Mean Intensities: {base_channel_means}")

    if len(metrics) == 0:
        print("No valid assays for comparison.")
        return None, None, None, None, None

    # overall_calibrated_avg = np.mean(metrics)
    # print(f"Overall Calibrated Grayscale Average: {overall_calibrated_avg}")

    result = cv2.addWeighted(croppedBaseImage, 0.6, main_background, 0.4, 0)

    return croppedBaseImage, croppedImage, main_background, result, metrics


def generateResultsBulk(input_folder="stand images", results_folder="results"):
    entries = []

    # Specify the base folder and the results folder
    input_folder = "/Users/apparilalith/Desktop/asu/mayolab/woundsensor/input_images/iphone"
    results_folder = "results"

    success_cnt = 0
    failure_cnt = 0
    # Create the results folder if it doesn't exist
    os.makedirs(results_folder, exist_ok=True)
    sample_dirs = [f for f in os.listdir(input_folder) if not (
        f.startswith('.') or f.endswith('.db'))]
    # Iterate over the sample folders
    for sample_name in sample_dirs:
        sample_folder = os.path.join(input_folder, sample_name)
        base_folder = os.path.join(input_folder, sample_name)
        dirs = [f for f in os.listdir(sample_folder) if not (
            f.startswith('.') or f.endswith('.db'))]
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
                        base_image_path = os.path.join(
                            base_dry_folder, filename)
                # Create the result folder path based on the image_path
                if image_path is None or base_image_path is None:
                    print("No file found : ", str(sample_type)+str(number))
                    continue
                relative_folder_path = os.path.dirname(image_path)
                relative_folder_path = os.path.relpath(
                    relative_folder_path, input_folder)
                result_folder_path = os.path.join(
                    results_folder, relative_folder_path)

                croppedBaseImage, croppedImage, main_background, result, metrics = calculateSimilarity(
                    base_image_path, image_path)
                if (metrics == None):
                    print(image_path)
                    failure_cnt += 1
                    continue
                success_cnt += 1
                # metrics has four entries for each circle, each circle has 2 measures number of pixels changed, intensity change
                print(sample_name, sample_type, metrics)
                # Add the entry to the DataFrame
                entry = {
                    'Concentration': sample_name,
                    'Sample Type': sample_type,
                    'Sample Number': number,
                    'Mean': metrics
                }
                entries.append(entry)

                # Create the result folder if it doesn't exist
                os.makedirs(result_folder_path, exist_ok=True)
                # Construct the result filepath
                result_filename = f"result_{sample_name}_{sample_type}_{number}.png"
                result_filepath = os.path.join(
                    result_folder_path, result_filename)

                # Display and save the figures
                plt.figure(figsize=(16, 10))
                plt.subplot(1, 4, 1)
                # plt.imshow(cv2.cvtColor(croppedBaseImage, cv2.COLOR_BGR2RGB))
                plt.imshow(cv2.cvtColor(cv2.imread(base_image_path,
                           cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                plt.title("Base Image")
                plt.axis("off")

                plt.subplot(1, 4, 2)
                plt.imshow(cv2.cvtColor(cv2.imread(
                    image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
                # plt.imshow(cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))
                plt.title("Image")
                plt.axis("off")

                plt.subplot(1, 4, 3)
                plt.imshow(cv2.cvtColor(main_background, cv2.COLOR_BGR2RGB))
                plt.title("Heatmap Difference")
                plt.axis("off")

                plt.subplot(1, 4, 4)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.imshow(cv2.cvtColor(main_background, cv2.COLOR_BGR2RGB),
                           cmap='jet', interpolation='none', alpha=0.35)
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
    if image_path is None or base_image_path is None:
        print("No file found  ")
        return None, None
    croppedBaseImage, croppedImage, main_background, result, metrics = calculateSimilarity(base_image_path, image_path)
    # print(metrics)

    if croppedBaseImage is None or croppedImage is None:
        print('Error while processing the images')
        return None, None
    base_image_height, base_image_width, _ = croppedBaseImage.shape
    image_height, image_width, _ = croppedImage.shape
    
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1.5, 1])

    # Display and save the figures
    # plt.figure(figsize=(16, 4))

    # Subplot 1: Base Image
    # plt.subplot(gs[0])
    # plt.imshow(cv2.cvtColor(croppedBaseImage, cv2.COLOR_BGR2RGB))
    # # plt.imshow(cv2.cvtColor(cv2.imread(base_image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    # plt.title("Base Image")
    # plt.axis("off")

    # # Subplot 2: Image
    # plt.subplot(gs[1])
    # plt.imshow(cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))
    # # plt.imshow(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    # plt.title("Image")
    # plt.axis("off")

    # # Subplot 3: Metrics for Circles
    # ax1 = plt.subplot(gs[2])
    # Extracting the circle numbers and corresponding values
    # circle_numbers = list(range(1, 5))  # Assuming there are 4 circles (adjust accordingly)
    # pixels_changed = [metrics[i][0] for i in range(4)]
    # intensity_change = [metrics[i][1] for i in range(4)]

    # Plotting pixels changed on the primary y-axis
    # ax1.plot(circle_numbers, pixels_changed, label='Pixels Changed', marker='o', color='tab:blue')
    # ax1.set_xlabel('Circle Number')
    # ax1.set_ylabel('Pixels Changed', color='tab:blue')

    # # Creating a secondary y-axis for intensity change
    # ax2 = ax1.twinx()
    # ax2.plot(circle_numbers, intensity_change, label='Intensity Change', marker='o', color='tab:red')
    # ax2.set_ylabel('Intensity Change', color='tab:red')

    # # Adding title
    # plt.title('Metrics for Circles')

    # # Set x-axis tick positions and labels to display as "Circle 1", "Circle 2", etc.
    # ax1.set_xticks(circle_numbers)
    # ax1.set_xticklabels([f'Circle {i}' for i in circle_numbers])

    # Adding legend
    # ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')

    # plt.show()

    # Subplot 4: Overlay with Colorbar
    # plt.subplot(gs[3])
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.imshow(cv2.cvtColor(main_background, cv2.COLOR_BGR2RGB), cmap='jet', interpolation='none', alpha=0.35)
    # plt.title("Overlay")
    # plt.axis("off")
    # cbar = plt.colorbar(aspect=40)  # Adjust the aspect ratio for the colorbar here
    # cbar.ax.tick_params(labelsize=8)

    # print(os.path.basename(image_path).split('.')[0])
    # # Adjust the layout and save the plot
    # plt.tight_layout()
    # result_file_path = os.path.join(result_filepath, 'results_' + str(os.path.basename(image_path).split('.')[0]) + '.png')
    # print(result_file_path)
    # plt.savefig(result_file_path)
    # plt.close()

    # plt.close()
    return croppedBaseImage, metrics
