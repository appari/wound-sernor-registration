import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2

def changeFormat(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def align_images(image, template, maxFeatures=100, keepPercent=0.2, debug=False):
    orb = cv2.ORB_create(maxFeatures)
    keypointsA, descriptorsA = orb.detectAndCompute(image, None)
    keypointsB, descriptorsB = orb.detectAndCompute(template, None)

    # Match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descriptorsA, descriptorsB, None)

    matches = sorted(matches, key=lambda x: x.distance)
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    if debug:
        image_kps = cv2.drawKeypoints(image, keypointsA, None, color=(0, 255, 0),
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        template_kps = cv2.drawKeypoints(template, keypointsB, None, color=(0, 255, 0),
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image_kps)
        axs[0].set_title('Image with Keypoints')
        axs[0].axis('off')
        axs[1].imshow(template_kps)
        axs[1].set_title('Template with Keypoints')
        axs[1].axis('off')
        plt.show()

        matchedVis = cv2.drawMatches(image, keypointsA, template, keypointsB, matches, None)
        matchedVis = cv2.resize(matchedVis, (1000, int(1000 * template.shape[0] / template.shape[1])))
        plt.figure(figsize=(8, 6))
        plt.imshow(matchedVis)
        plt.show()

    ptsA = np.zeros((len(matches), 2), dtype="float32")
    ptsB = np.zeros((len(matches), 2), dtype="float32")
    for i, m in enumerate(matches):
        ptsA[i] = keypointsA[m.queryIdx].pt
        ptsB[i] = keypointsB[m.trainIdx].pt

    # Estimate affine transformation
    M, _ = cv2.estimateAffinePartial2D(ptsA, ptsB)

    # Apply affine transformation to align the images
    aligned = cv2.warpAffine(image, M, (template.shape[1], template.shape[0]))

    return aligned, M

def find_rectangle_contours(contours):
    rectangles = []

    # Iterate through all combinations of four contours
    for i in range(len(contours) - 3):
        for j in range(i + 1, len(contours) - 2):
            for k in range(j + 1, len(contours) - 1):
                for l in range(k + 1, len(contours)):
                    # Calculate the distances between adjacent centroids
                    centroids = [
                        np.array(cv2.mean(contours[i])[:2]),
                        np.array(cv2.mean(contours[j])[:2]),
                        np.array(cv2.mean(contours[k])[:2]),
                        np.array(cv2.mean(contours[l])[:2])
                    ]
                    distances = [
                        cv2.norm(centroids[0] - centroids[1]),
                        cv2.norm(centroids[1] - centroids[2]),
                        cv2.norm(centroids[2] - centroids[3]),
                        cv2.norm(centroids[3] - centroids[0])
                    ]

                    # Check if the distances between adjacent centroids are approximately equal
                    avg_distance = sum(distances) / len(distances)
                    max_deviation = max([abs(dist - avg_distance) for dist in distances])

                    # Adjust the deviation threshold based on your requirements
                    deviation_threshold = 0.5 * avg_distance
                    if max_deviation < deviation_threshold:
                        # Check if the centroids are approximately circular and have the correct distance
                        radius = 0.5 * max([cv2.boundingRect(contour)[2] for contour in [contours[i], contours[j], contours[k], contours[l]]])
                        tolerance = 1 * radius
                        check = [abs(dist - 2 * radius) < tolerance for dist in distances]
                        circular_check = any(check)
                        # print(check)
                        if circular_check:
                            # Append the corresponding four contours to the rectangles list
                            rectangles.append([contours[i], contours[j], contours[k], contours[l]])
    if(len(rectangles)>0):
        return rectangles[0]
    return []

def filter_contours_by_centroids(contours):
    filtered_contours = []
    centroids = []

    # Extract centroids and calculate bounding boxes
    for contour in contours:
        moments = cv2.moments(contour)
        centroid_x = int(moments['m10'] / moments['m00'])
        centroid_y = int(moments['m01'] / moments['m00'])
        centroids.append((centroid_x, centroid_y))

    for i in range(len(contours)):
        contour = contours[i]
        centroid = centroids[i]

        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)
        radius = 2 * w

        # Check if any other centroid lies within the circle
        is_valid = False
        for j in range(len(contours)):
            if j != i:
                other_centroid = centroids[j]
                distance = np.sqrt((centroid[0] - other_centroid[0]) ** 2 + (centroid[1] - other_centroid[1]) ** 2)
                if distance <= radius:
                    is_valid = True
                    break

        if is_valid:
            filtered_contours.append(contour)

    return filtered_contours

def filter_blob_contours(contours):
    filtered_contours = []

    for contour in contours:
        # Calculate the area and perimeter of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Calculate the compactness (circularity) of the contour
        compactness = 4 * np.pi * area / (perimeter ** 2)

        # Filter contours based on compactness threshold
        if compactness > 0.5:  # Adjust the threshold as needed
            filtered_contours.append(contour)

    return filtered_contours


def extract_yellowcircles(image, padding_ratio = 0.9):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([45, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    yellow_circles = cv2.bitwise_and(image, image, mask=closing)
    gray = cv2.cvtColor(yellow_circles, cv2.COLOR_BGR2GRAY)

    # Apply Difference of Gaussians (DoG) to enhance blob-like structures
    sigma1 = 1.5
    sigma2 = 3.0
    ksize = int(2 * round(3 * sigma1) + 1)
    blur1 = cv2.GaussianBlur(gray, (ksize, ksize), sigma1)
    blur2 = cv2.GaussianBlur(gray, (ksize, ksize), sigma2)
    dog = blur1 - blur2

    threshold = 100
    _, binary = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)
    # _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # plt.figure()
    # plt.imshow(yellow_circles)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(yellow_circles, contours, 2, 255, 10)
    if(len(contours)>4):
        contours = filter_blob_contours(contours)
        contours = filter_contours_by_centroids(contours)
        if(len(contours)!=4):
            contours = find_rectangle_contours(contours)


    # filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    threshold_area = 0
    lower_yellow = np.array([15, 100, 80])
    upper_yellow = np.array([45, 255, 255])

    yellow_objects = []
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    max_width = -1
    areas = []
    # TODO: Add validation check to eliminate noisy contours using geometry of the sensor
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > threshold_area:
            mask = np.zeros_like(hsv[:,:,0], dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_color = cv2.mean(hsv, mask=mask)[:3]  # HSV format
            if (lower_yellow <= mean_color).all() and (mean_color <= upper_yellow).all():
                areas.append(area)
                x, y, w, h = cv2.boundingRect(contour)
                max_width = max(max_width, max(w,h))
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)
                M = cv2.moments(contour)
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])
                yellow_objects.append((centroid_x, centroid_y, [contour]))
    if(len(yellow_objects)==0):
        return yellow_circles, image, max_width/2, image, 0, []
    # for (x, y) in yellow_objects:
    #     cv2.circle(image, (x, y), 2, (0, 255, 255), -1)
    # print((x_max - x_min) * padding_ratio)
    padding = int((x_max - x_min) * padding_ratio)
    x_min = max(x_min - padding, 0)
    y_min = max(y_min - padding, 0)
    x_max = min(x_max + padding, image.shape[1])
    y_max = min(y_max + padding, image.shape[0])

    square_region = image[y_min:y_max, x_min:x_max]
    return yellow_circles, square_region, max_width/2, image, np.max(areas), yellow_objects
