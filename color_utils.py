import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2
import math

def getFinalPts(black_objs, white_objs, yellow_objs):
    # print('black objs', black_objs)
    # print('white objs', white_objs)
    # print('yellow objs', yellow_objs)
    def rearrange_quadrilateral_points(points):
        # Find the centroid of the quadrilateral
        centroid_x = sum(point[0] for point in points) / len(points)
        centroid_y = sum(point[1] for point in points) / len(points)
        centroid = (centroid_x, centroid_y)

        # Sort the points based on their angle relative to the centroid
        sorted_points = sorted(points, key=lambda point: math.atan2(point[1] - centroid[1], point[0] - centroid[0]))
        return  np.array([(x,y) for x,y in sorted_points], dtype=np.int32)

    def filterPts(objs, yellow_objs):
        if(len(objs)==0):
            return []
        pts = np.array([(x,y) for x,y,b in yellow_objs], dtype=np.int32)
        pts = rearrange_quadrilateral_points(pts)
        pts = pts.reshape((-1, 1, 2))
        final = []
        for pt in objs:
            if cv2.pointPolygonTest(pts, pt, False) < 0:
                final.append(pt)
        return final
    black_objs = filterPts(black_objs, yellow_objs)
    white_objs = filterPts(white_objs, yellow_objs)

    min_dist = float('inf')
    min_ypt = None
    min_wpt = None
    obj = white_objs
    if(len(white_objs)==0):
        obj = black_objs
    for wx,wy in obj:
        for yx, yy, bb in yellow_objs:
            dist = math.sqrt((wx - yx)**2 + (wy - yy)**2)
            if(dist < min_dist):
                min_dist = dist
                min_ypt = (yx, yy, bb)
                min_wpt = (wx, wy)
    # print(min_wpt, min_ypt)
    def distance_to_line(point, line_point1, line_point2):
        x, y = point
        x1, y1 = line_point1
        x2, y2 = line_point2
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return numerator / denominator

    second_mindist = float('inf')
    second_minypt = None

    if(min_ypt == None):
        return None, None
    for yx, yy, bb in yellow_objs:
        if yx == min_ypt[0] and yy == min_ypt[1] :
            continue
        distance = distance_to_line((yx, yy), min_wpt, min_ypt[:2])
        # print(distance, yx, yy)
        if distance < second_mindist:
            second_mindist = distance
            second_minypt = (yx, yy, bb)

    top_pts =[min_ypt, second_minypt]
    bottom_pts = []

    for yx, yy, bb in yellow_objs:
        if (yx, yy, bb) not in top_pts:
            bottom_pts.append((yx, yy, bb))
    
    # print(top_pts)
    if len(white_objs)==0:
        top_pts[:], bottom_pts[:] = bottom_pts[:], top_pts[:]
        
    def calculate_cross_product(p1, p2, p3):
        # Calculate the cross product of vectors (p2 - p1) and (p3 - p1)
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        return cross_product
    final_order = [None]*4
    if(len(bottom_pts)<2 or len(top_pts)<2):
        return None, None
    if(calculate_cross_product(top_pts[0], top_pts[1], bottom_pts[0]) > 0):
        final_order[0] = top_pts[0]
        final_order[1] = top_pts[1]
    else:
        final_order[0] = top_pts[1]
        final_order[1] = top_pts[0]
    
    if(calculate_cross_product(bottom_pts[0], bottom_pts[1], top_pts[0]) > 0):
        final_order[2] = bottom_pts[0]
        final_order[3] = bottom_pts[1]
    else:
        final_order[2] = bottom_pts[1]
        final_order[3] = bottom_pts[0]

    return final_order


# Define color thresholds for black, white, and yellow
color_thresholds = {
    'black': {
        'lower_hsv': np.array([0, 0, 0]),
        'upper_hsv': np.array([179, 255, 50]),
        'color_threshold': 70
    },
    'white': {
        'lower_hsv': np.array([0, 0, 200]),
        'upper_hsv': np.array([179, 50, 255]),
        'color_threshold': 150
    },
    'yellow': {
        'lower_hsv': np.array([20, 100, 100]),
        'upper_hsv': np.array([40, 255, 255]),
        'color_threshold': 120
    }
}

def extractColoredBoundingBox(im, color, yellowArea):
    lower_hsv = color_thresholds[color]['lower_hsv']
    upper_hsv = color_thresholds[color]['upper_hsv']

    color_threshold = color_thresholds[color]['color_threshold']

    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Apply Difference of Gaussians (DoG) to enhance blob-like structures
    sigma1 = 1.5
    sigma2 = 3.0
    ksize = int(2 * round(3 * sigma1) + 1)
    blur1 = cv2.GaussianBlur(mask, (ksize, ksize), sigma1)
    blur2 = cv2.GaussianBlur(mask, (ksize, ksize), sigma2)
    dog = blur1 - blur2

    threshold = 80
    _, binary = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)
    plt.figure()
    # plt.imshow(binary)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, contours, -1, (0, 255, 0), 2)
    scaleFactor = 2
    threshold_area = 0
    round_objects = []
    # print(len(contours))
    for i, c in enumerate(contours):
        boundRect = cv2.boundingRect(c)
        rectX = int(boundRect[0])
        rectY = int(boundRect[1])
        rectWidth = int(boundRect[2])
        rectHeight = int(boundRect[3])
        contourArea = cv2.contourArea(c)

        referenceRatio = 1.0
        contourRatio = rectWidth / rectHeight
        epsilon = 1.1
        ratioDifference = abs(referenceRatio - contourRatio)

        centroids = []
        # Blue: Filtered
        color = (0, 0, 255)
        minArea = yellowArea * 0.4
        maxArea = 1.5 * yellowArea
        # print(yellowArea)
        if ratioDifference <= epsilon:
            if minArea <= contourArea < maxArea:
                mask = np.zeros_like(mask)
                # cv2.drawContours(mask, [c], 0, 255, -1)
                mean_color = cv2.mean(hsv, mask=mask)[:3]  # HSV format
                color_check = True
                if color == 'white' and mean_color[2] < color_threshold:
                    color_check = False
                if color == 'black' and mean_color[2] > color_threshold:
                    color_check = False
                if color_check:
                    M = cv2.moments(c)
                    centroid_x = int(M['m10'] / M['m00'])
                    centroid_y = int(M['m01'] / M['m00'])

                    round_objects.append((centroid_x, centroid_y))
                    croppedChar = im[rectY:rectY + rectHeight, rectX:rectX + rectWidth]
                    # plt.figure()
                    # plt.imshow(croppedChar)
                else:
                    print("Color Check failed", mean_color)
                # Green: Detected bounding box
                color = (0, 255, 0)
            else:
                pass
                # print("Area Check failed", contourArea, minArea, maxArea, yellowArea)
        # for (x,y) in round_objects:
            # cv2.circle(cropped_im, (x, y), 3, (0, 255, 0), 2)
        # cv2.rectangle(im, (int(rectX), int(rectY)),
        #               (int(rectX + rectWidth), int(rectY + rectHeight)), color, 2)

    return im, round_objects

def getHeatMap(image, contour):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_gray = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap((normalized_gray * 255).astype(np.uint8), cv2.COLORMAP_JET)

    contour_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], 0, 255, -1)
    masked_heatmap = cv2.bitwise_and(heatmap, heatmap, mask=contour_mask)
    # result = masked_heatmap
    result = cv2.addWeighted(image, 0.7, masked_heatmap, 0.3, 0)

    # Display the result
    # plt.figure(figsize=(10,8))
    # plt.imshow(result[:,:,::-1])  
    # plt.axis('off')
    # plt.show()
    return result

def getHeatMapColorDifference(image1, image2, normalization='min-max'):
    # Compute the color difference between the RGB images
    color_diff = cv2.absdiff(image1, image2)

    # Sum the color difference across channels
    color_diff_sum = np.sum(color_diff, axis=2)

    # Normalize the color difference values
    if normalization == 'min-max':
        normalized_diff = cv2.normalize(color_diff_sum, None, 0, 255, cv2.NORM_MINMAX)
    elif normalization == 'adaptive':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_diff = clahe.apply(color_diff_sum.astype(np.uint8))
    else:
        normalized_diff = color_diff_sum

    # Create a heatmap using the normalized color difference values
    heatmap = cv2.applyColorMap(normalized_diff.astype(np.uint8), cv2.COLORMAP_JET)

    return heatmap

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled
