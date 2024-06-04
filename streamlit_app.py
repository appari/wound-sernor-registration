import streamlit as st
import cv2
from PIL import Image
import numpy as np
from utils import generateResultsSingleImage
import os

def process_images(image1, image2):
    img1_np = np.array(image1)
    img2_np = np.array(image2)

    result = cv2.add(img1_np, img2_np)

    return result

def save_uploaded_files_to_temp(uploaded_files):
    temp_folder = "temp"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)

    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return file_paths

def main():
    st.title("Wound sensor image processing")

    uploaded_image1 = st.file_uploader("Upload Base Image", type=['jpg', 'png', 'jpeg', 'DNG'])
    uploaded_image2 = st.file_uploader("Upload Exposed Image", type=['jpg', 'png', 'jpeg', 'DNG'])

    if uploaded_image1 is not None and uploaded_image2 is not None:
        st.image(uploaded_image1, caption='Upload Base Image', use_column_width=True)
        st.image(uploaded_image2, caption='Upload Exposed Image', use_column_width=True)

        if st.button('Process Images'):
            image_paths = save_uploaded_files_to_temp([uploaded_image1, uploaded_image2])
            # print(image_paths[0])
            result_image_path = generateResultsSingleImage(image_paths[0], image_paths[1], '.')
            # print("result,image", result_image_path)
            if result_image_path:   
                result_image_pil = Image.open(result_image_path)
                st.image(result_image_pil, caption='Result Image')
            else:
                st.write("Error occurred during image processing.")

if __name__ == '__main__':
    main()