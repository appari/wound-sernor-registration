import numpy as np
import imutils
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

def drawBoxPlotsIntensity(df):
    # Grouping and Aggregation: Mean Intensity Change by Concentration for each circle
    mean_pixels_circle1 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 1 Intensity Change'].mean().reset_index()
    mean_pixels_circle2 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 2 Intensity Change'].mean().reset_index()
    mean_pixels_circle3 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 3 Intensity Change'].mean().reset_index()
    mean_pixels_circle4 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 4 Intensity Change'].mean().reset_index()

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    # Plot boxplots for Circle 1
    sns.boxplot(x='Concentration-val', y='Circle 1 Intensity Change', data=mean_pixels_circle1, ax=axes[0, 0])
    axes[0, 0].set_title('Circle 1')
    axes[0, 0].set_xlabel('Concentration')
    axes[0, 0].set_ylabel('Intensity Change')

    # Plot boxplots for Circle 2
    sns.boxplot(x='Concentration-val', y='Circle 2 Intensity Change', data=mean_pixels_circle2, ax=axes[0, 1])
    axes[0, 1].set_title('Circle 2')
    axes[0, 1].set_xlabel('Concentration')
    axes[0, 1].set_ylabel('Intensity Change')

    # Plot boxplots for Circle 3
    sns.boxplot(x='Concentration-val', y='Circle 3 Intensity Change', data=mean_pixels_circle3, ax=axes[1, 0])
    axes[1, 0].set_title('Circle 3')
    axes[1, 0].set_xlabel('Concentration')
    axes[1, 0].set_ylabel('Intensity Change')

    # Plot boxplots for Circle 4
    sns.boxplot(x='Concentration-val', y='Circle 4 Intensity Change', data=mean_pixels_circle4, ax=axes[1, 1])
    axes[1, 1].set_title('Circle 4')
    axes[1, 1].set_xlabel('Concentration')
    axes[1, 1].set_ylabel('Intensity Change')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def drawBoxPlotsPixelsPercentageChange(df):
    # Grouping and Aggregation: Mean Pixels Changed by Concentration for each circle
    mean_pixels_circle1 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 1 Pixels Changed'].mean().reset_index()
    mean_pixels_circle2 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 2 Pixels Changed'].mean().reset_index()
    mean_pixels_circle3 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 3 Pixels Changed'].mean().reset_index()
    mean_pixels_circle4 = df.groupby(['Concentration-val', 'Sample Type', 'Sample Number'])['Circle 4 Pixels Changed'].mean().reset_index()

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    # Plot boxplots for Circle 1
    sns.boxplot(x='Concentration-val', y='Circle 1 Pixels Changed', data=mean_pixels_circle1, ax=axes[0, 0])
    axes[0, 0].set_title('Circle 1')
    axes[0, 0].set_xlabel('Concentration')
    axes[0, 0].set_ylabel('Pixels Changed')

    # Plot boxplots for Circle 2
    sns.boxplot(x='Concentration-val', y='Circle 2 Pixels Changed', data=mean_pixels_circle2, ax=axes[0, 1])
    axes[0, 1].set_title('Circle 2')
    axes[0, 1].set_xlabel('Concentration')
    axes[0, 1].set_ylabel('Pixels Changed')

    # Plot boxplots for Circle 3
    sns.boxplot(x='Concentration-val', y='Circle 3 Pixels Changed', data=mean_pixels_circle3, ax=axes[1, 0])
    axes[1, 0].set_title('Circle 3')
    axes[1, 0].set_xlabel('Concentration')
    axes[1, 0].set_ylabel('Pixels Changed')

    # Plot boxplots for Circle 4
    sns.boxplot(x='Concentration-val', y='Circle 4 Pixels Changed', data=mean_pixels_circle4, ax=axes[1, 1])
    axes[1, 1].set_title('Circle 4')
    axes[1, 1].set_xlabel('Concentration')
    axes[1, 1].set_ylabel('Pixels Changed')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

def singleImageScatterPlot(df):
    # Assuming you have a DataFrame called 'df' with columns 'Sample Name', 'Circle 1 Intensity Change', 'Circle 2 Intensity Change', 'Circle 3 Intensity Change', 'Circle 4 Intensity Change'

    # Create a dictionary mapping concentration values to corresponding numeric values
    concentration_mapping = {'0mM': 0, '10mM': 10, '20mM': 20, '30mM': 30}

    # Create a new column 'Concentration-val' based on the mapping
    df['Concentration-val'] = df['Concentration'].map(concentration_mapping)

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot intensity change for each circle in a subplot
    axs[0, 0].scatter(df['Concentration-val'], df['Circle 1 Pixels Changed'])
    axs[0, 0].set_xlabel("Concentration")
    axs[0, 0].set_ylabel("Pixels Changed - Circle 1")
    axs[0, 0].set_title("Circle 1")

    axs[0, 1].scatter(df['Concentration-val'], df['Circle 2 Pixels Changed'])
    axs[0, 1].set_xlabel("Concentration")
    axs[0, 1].set_ylabel("Pixels Changed - Circle 2")
    axs[0, 1].set_title("Circle 2")

    axs[1, 0].scatter(df['Concentration-val'], df['Circle 3 Pixels Changed'])
    axs[1, 0].set_xlabel("Concentration")
    axs[1, 0].set_ylabel("Pixels Changed - Circle 3")
    axs[1, 0].set_title("Circle 3")

    axs[1, 1].scatter(df['Concentration-val'], df['Circle 4 Pixels Changed'])
    axs[1, 1].set_xlabel("Concentration")
    axs[1, 1].set_ylabel("Pixels Changed - Circle 4")
    axs[1, 1].set_title("Circle 4")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()