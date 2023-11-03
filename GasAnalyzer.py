#!/usr/bin/env python
# coding: utf-8

# In[7]:


from openpyxl import load_workbook
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import font
import pandas as pd
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import seaborn as sns
import math
from functools import partial
import numpy as np
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from PIL import Image, ImageTk

def resource_path(relative_path):
    #""" Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
                        
def reset_globals():
    global pca_result_window, result_frame, gas_result_window
    pca_result_window = None
    result_frame = None
    gas_result_window = None
    
# Create a new tkinter window for displaying results
gas_result_window = None

# Create a new tkinter window for displaying PCA results
pca_result_window = None

# Create a global variable for the result_frame
result_frame = None

# Create a global variable for the canvas
canvas_gas = None

def update_scroll_region_gas(event):
    global canvas_gas

    # Get the underlying Tkinter canvas
    tk_canvas = canvas_gas.get_tk_widget()
    # Configure the scroll region based on the bounding box of all items on the canvas
    tk_canvas.configure(scrollregion=tk_canvas.bbox("all"))
    # Perform vertical scrolling based on the mouse wheel movement
    tk_canvas.yview_moveto(0)
    tk_canvas.yview_scroll(-1 * int((event.delta / 120)), "units")


def open_Gfile_dialog():
    # Open the file dialog box
    file_path = filedialog.askopenfilename()

    # Process the selected file
    if file_path != '':
        Gas(file_path, pd)

def open_Pfile_dialog():
    # Open the file dialog box
    file_path = filedialog.askopenfilename()

    # Process the selected file
    if file_path != '':
        perform_PCA(file_path, pd)

def open_Gas_frame():
    # Hide the current frame
    main_frame.pack_forget()

    # Show the new frame with the "Open File" button
    Gas_frame.pack()
    
def goP_Home():
    #Hide PCA Frame
    PCA_frame.pack_forget()
    
    #Show main frame
    main_frame.pack()
    
def goG_Home():
    #Hide PCA Frame
    Gas_frame.pack_forget()
    
    #Show main frame
    main_frame.pack()

def open_PCA_frame():
    # Hide the current frame
    main_frame.pack_forget()

    # Show the new frame with the "Open File" button
    PCA_frame.pack()

def perform_PCA_with_user_input():
    # Get the user input from the textbox
    gases_input = gases_entry.get()
    # Split the input into individual gas names
    target_gases = [gas.strip() for gas in gases_input.split(",")]

    # Open the file dialog box to select the Excel file
    file_path = filedialog.askopenfilename()

    # Process the selected file
    if file_path != '':
        # Create the pca_result_window if it doesn't exist
        global pca_result_window
        if pca_result_window is None:
            pca_result_window = tk.Toplevel()
            pca_result_window.title("PCA Results")
            pca_result_window.geometry(f"{window.winfo_screenwidth()//2}x{window.winfo_screenheight()//2}")
            pca_result_window.protocol("WM_DELETE_WINDOW", close_pca_result_window)  # Handle window close event

        # Call the perform_PCA function with the correct file path and pca_result_window
        perform_PCA(file_path, pd, target_gases, pca_result_window)
    else:
        # Show an error message when no file is selected
        tk.messagebox.showerror("Error", "No file selected for PCA.")
    
def perform_Gas_with_user_input():
    #pdb.set_trace()
    # Get the user input from the textbox
    sensors_input = sensors_entry.get()
    # Split the input into individual gas names
    target_sensors = [sensor.strip() for sensor in sensors_input.split(",")]

    # Open the file dialog box to select the Excel file
    file_path = filedialog.askopenfilename()

    # Process the selected file
    if file_path != '':
        # Create the gas_result_window if it doesn't exist
        global gas_result_window
        if gas_result_window is None:
            gas_result_window = tk.Toplevel()
            gas_result_window.title("Gas Response Results")
            gas_result_window.geometry(f"{window.winfo_screenwidth()//2}x{window.winfo_screenheight()//2}")
            gas_result_window.protocol("WM_DELETE_WINDOW", close_gas_result_window)  # Handle window close event

        # Call the open_Gfile_dialog function with the correct file path and target_sensors
        Gas(file_path, target_sensors)
    else:
        # Show an error message when no file is selected
        tk.messagebox.showerror("Error", "No file selected for Gas Response.")    
    
def close_pca_result_window():
    global pca_result_window
    global result_frame

    # Destroy the result_frame if it exists
    if result_frame is not None:
        result_frame.destroy()

    # Destroy the pca_result_window if it exists
    if pca_result_window is not None:
        pca_result_window.destroy()

    # Reset the result_frame and pca_result_window to None
    result_frame = None
    pca_result_window = None

def close_gas_result_window():
    global gas_result_window
    global result_frame

    # Destroy the result_frame if it exists
    if result_frame is not None:
        result_frame.destroy()

    # Destroy the gas_result_window if it exists
    if gas_result_window is not None:
        gas_result_window.destroy()

    # Reset the result_frame and gas_result_window to None
    result_frame = None
    gas_result_window = None
    
def display_pca_results(fig, pca_result_window):
    #global pca_result_window  # Add this line to access the global variable

    # Create the pca_result_window if it doesn't exist
    if pca_result_window is None:
        pca_result_window.title("PCA Results")
        pca_result_window.geometry(f"{window.winfo_screenwidth()//2}x{window.winfo_screenheight()//2}")
        #pca_result_window.withdraw()
        pca_result_window.protocol("WM_DELETE_WINDOW", close_pca_result_window)  # Handle window close event

    # Clear the existing content in the window
    for widget in pca_result_window.winfo_children():
        widget.destroy()

    # Create a frame to hold the Figure and the toolbar
    frame = ttk.Frame(pca_result_window)
    frame.pack(fill=tk.BOTH, expand=True)

    # Create a FigureCanvasTkAgg instance
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create a NavigationToolbar2Tk instance
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # Make the pca_result_window visible
    pca_result_window.deiconify()
    
def display_gas_results(fig, gas_result_window):
    #global pca_result_window  # Add this line to access the global variable

    # Create the pca_result_window if it doesn't exist
    #if gas_result_window is None:
        #gas_result_window.title("Gas Results")
        #gas_result_window.geometry(f"{window.winfo_screenwidth()//2}x{window.winfo_screenheight()//2}")
        #pca_result_window.withdraw()
        #gas_result_window.protocol("WM_DELETE_WINDOW", close_gas_result_window)  # Handle window close event

    # Clear the existing content in the window
    for widget in gas_result_window.winfo_children():
        widget.destroy()

    # Create a frame to hold the Figure and the toolbar
    #frame = ttk.Frame(gas_result_window)
    #frame.pack(fill=tk.BOTH, expand=True)

    # Create a FigureCanvasTkAgg instance
    #canvas = FigureCanvasTkAgg(fig, master=frame)
    #canvas.draw()
    #canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Create a NavigationToolbar2Tk instance
    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # Make the pca_result_window visible
    gas_result_window.deiconify()

def perform_PCA(file_path, pd, target_gases, pca_result_window):
    #global pca_result_window  # Use the global pca_result_window variable
    # Read the Excel file and skip the header row
    worksheet = pd.read_excel(file_path, header=None)

    # Define the column names in the desired order
    column_names = [0, 1, 2, 3, 'type', 'day']

    # Assign the column names to the DataFrame
    df = worksheet.copy()
    df.columns = column_names

    features = [0, 1, 2, 3]  # Use numerical column indices instead of column names
    x = df.iloc[:, features].values  # Use iloc to slice the DataFrame by numerical indices
    y = df.loc[:,['type']].values
    
    #calibration = ['1', '3']
    calibration = ['1', str(len(target_gases))]
    #if target_gases is None:
        #target_gases = ['Hydrogen', 'Isopropanol', 'Acetone']
    for i in features: #Iterating through each sensor
        data = df[[i, 'type', 'day']]
        corrected = []
        gases = []
        for gas in target_gases: #Iterating through each gas
            filtered = []
            for j in range(len(df)): #Loop to choose only the data with the sensor and gas
                if df.loc[j, 'type'] == gas:
                    filtered.append(df.loc[j, i])
            q = []
            for j in range(len(calibration)): #Calculates the quotients for each sensor and gas
                q.append(float(filtered[int(calibration[0])-1])/float(filtered[int(calibration[j])-1]))
            x = np.arange(0,len(q))
            coeff = np.polyfit(x, q, 1) #Finds the line of best fit (change last parameter to change degree of best fit)
            index = 0
            for j in range(len(filtered)): #Modifies the data based on a linear correction
                if index < len(calibration) and int(calibration[index])-1 == j:
                    index = index + 1
                if index == len(calibration) and int(calibration[index-1])-1 == j: #If the calibration is the last data taken
                    corrected.append(float(filtered[j])*(coeff[0]*(index-1)+coeff[1])) 
                elif index == len(calibration): #If the index is past the last calibration point
                    corrected.append(float(filtered[j])*(coeff[0]*(index-1+(j-int(calibration[index-1]))/(len(filtered)-int(calibration[index-1])))+coeff[1])) 
                else: #If neither of the other two apply
                    corrected.append(float(filtered[j])*(coeff[0]*(index-1+(j-int(calibration[index-1])+1)/(int(calibration[index])-int(calibration[index-1])))+coeff[1])) 

                #corrected.append(float(filtered[j])*(coeff[0]*j*0.5+coeff[1])) #change x_n depending on the calibration
                gases.append(gas)
        z = int(i)
        df[df.columns[z]] = corrected
        if int(i) == len(features)-1:
            df['type'] = gases
    x = df.loc[:, features].values
    y = df.loc[:, 'type'].values

    x = StandardScaler().fit_transform(x)
    scaled_data = pd.DataFrame(x, columns=features)

    # Create subplots for the heatmaps and the plot
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    # Plot the correlation heatmap of scaled data
    sns.heatmap(scaled_data.corr(), annot=True, ax=axes[0])
    axes[0].set_title('Correlation Heatmap')

    # Perform PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    
    # Create a new Toplevel window to show the PCA results
    new_pca_result_window = tk.Toplevel()

    # Plot the correlation heatmap after PCA
    sns.heatmap(principalDf.corr(), annot=True, ax=axes[1])
    axes[1].set_title('Correlation Heatmap after PCA')

    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # Plot the explained variance graph
    axes[2].bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    axes[2].step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
    axes[2].set_ylabel('Explained variance ratio')
    axes[2].set_xlabel('Principal component index')
    axes[2].legend(loc='best')
    axes[2].set_title('Explained Variation Graph')

    # Adjust the spacing between subplots
    plt.tight_layout()

    total_var = pca.explained_variance_ratio_.sum() * 100
    pcaDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    y = df.loc[:, 'type'].values
    pcaDf = pcaDf.assign(gasType=y)

    colors = ['navy', 'turquoise', 'darkorange', 'red']

    plt.figure(figsize=(10, 8))
    axes[3].set_title('2D PCA: 4 Sensor Array, Total Explained Variance: %2.2f' % total_var)
    axes[3].set_ylabel #plt.ylabel('PC2')
    axes[3].set_xlabel #plt.xlabel('PC1')

    for color, gas in zip(colors, target_gases):
        axes[3].scatter(pcaDf[pcaDf.gasType == gas].PC1, pcaDf[pcaDf.gasType == gas].PC2,
                    color=color, alpha=0.8, lw=2, s=4, label=gas)
        hull = ConvexHull(pcaDf[pcaDf.gasType == gas][['PC1', 'PC2']].values)
        for simplex in hull.simplices:
            axes[3].plot(pcaDf[pcaDf.gasType == gas].values[simplex, 0], pcaDf[pcaDf.gasType == gas].values[simplex, 1], color=color)

    axes[3].legend(loc='best', shadow=False, scatterpoints=1)
    #plt.legend(loc='best', shadow=False, scatterpoints=1)

    # Create the FigureCanvasTkAgg instance
    canvas = FigureCanvasTkAgg(fig, master=pca_result_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Show the plot
    display_pca_results(fig, new_pca_result_window)  # Pass the new_pca_result_window to display_pca_results
    
def Gas(file_path, target_sensors):
    global gas_result_window
    global result_frame

    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Get unique sensor names
    sensors = df.columns[1:]  # Exclude the 'Time' column

    # Set Seaborn style
    sns.set_style("whitegrid")

    # Create the gas_result_window if it doesn't exist
    if gas_result_window is None:
        gas_result_window = ttk.Toplevel()
        gas_result_window.title("Gas Results")
        gas_result_window.protocol("WM_DELETE_WINDOW", close_gas_result_window)  # Handle window close event

        gas_result_window.withdraw()  # Hide the window initially

        # Create a label for displaying Gas information
        gas_info_label = ttk.Label(gas_result_window, text="Gas Information:")
        gas_info_label.pack()

        # Create a canvas for scrolling
        canvas = ttk.Canvas(gas_result_window)
        canvas.pack(side=ttk.LEFT, fill=ttk.BOTH, expand=True)

        # Create a vertical scrollbar
        scrollbar = ttk.Scrollbar(gas_result_window, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=ttk.RIGHT, fill=ttk.Y)

        # Configure the canvas and scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)

        # Function to update the scrollbar region
        def update_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the canvas
        result_frame = ttk.Frame(canvas)
        # Add the frame to the canvas
        canvas.create_window((0, 0), window=result_frame, anchor='nw')

        # Bind the frame to the <Configure> event to update the scrollbar region
        result_frame.bind("<Configure>", update_scroll_region)

        # Allow scrolling with mouse wheel
        gas_result_window.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * int((e.delta / 120)), "units"))

    else:
        # Remove any existing widgets in the result_frame
        if result_frame is not None:
            for child in result_frame.winfo_children():
                child.destroy()

    # Iterate over each sensor
    for i, sensor in enumerate(sensors):
        # Get sensor-specific data
        time = df['Time']
        resistance = df[sensor]

        # Fill in missing points by connecting the two points around it
        resistance = resistance.interpolate(method='linear')

        # Calculate maximum resistance and powers of ten
        max_resistance = resistance.max()
        powers_of_ten = math.floor(math.log10(max_resistance))

        # Initialize variables for the first while loop
        index = 1
        diff_threshold = 1 * 10 ** (powers_of_ten - 1)
        sum_resistance = 0
        resistance_values_first = []

        # Perform the first while loop
        while index < len(resistance) - 3:
            diff = resistance.iloc[index] - resistance.iloc[index-1]
            unless_condition = resistance.iloc[index-1] - resistance.iloc[index-2] > diff_threshold
            if unless_condition:
                index += 1
                continue
            elif diff < -diff_threshold:
                break
            sum_resistance += resistance.iloc[index]
            resistance_values_first.append(resistance.iloc[index])
            index += 1

        # Calculate the average for the first while loop
        average_first = sum_resistance / len(resistance_values_first)

        # Initialize variables for the second while loop
        index2 = index + 1
        diff2_threshold = 0.1 * 1 * 10 ** (powers_of_ten - 1)
        diff2 = resistance.iloc[index2] - resistance.iloc[index2-1]

        # Perform the second while loop
        while index2 < len(resistance) - 1 and abs(diff2) > diff2_threshold:
            index2 += 1
            diff2 = resistance.iloc[index2] - resistance.iloc[index2-1]

        # Initialize variables for the third while loop
        index3 = index2 + 1
        diff3_threshold = 1 * 10 ** (powers_of_ten - 1)
        sum_resistance_third = 0
        resistance_values_third = []

        # Perform the third while loop
        while index3 < len(resistance) - 3:
            diff3 = resistance.iloc[index3] - resistance.iloc[index3-1]
            unless_condition3 = resistance.iloc[index3-1] - resistance.iloc[index3-2] > diff3_threshold
            if unless_condition3:
                index3 += 1
                continue
            elif diff3 > diff3_threshold:
                break
            sum_resistance_third += resistance.iloc[index3]
            resistance_values_third.append(resistance.iloc[index3])
            index3 += 1

        # Calculate the average for the third while loop
        average_third = sum_resistance_third / len(resistance_values_third)

        # Initialize variables for the fourth while loop
        index4 = index3 + 1
        diff4_threshold = 0.1 * 1 * 10 ** powers_of_ten
        diff4 = resistance.iloc[index4] - resistance.iloc[index4-1]
        sum_resistance_fourth = 0

        # Perform the fourth while loop
        while index4 < len(resistance) - 1 and abs(diff4) > diff4_threshold:
            index4 += 1
            diff4 = resistance.iloc[index4] - resistance.iloc[index4-1]
            sum_resistance_fourth += resistance.iloc[index4]

        # Initialize variables for the fifth while loop
        index5 = index4 + 1
        sum_resistance_fifth = 0
        resistance_values_fifth = []

        # Perform the fifth while loop
        while index5 < len(resistance) - 1:
            sum_resistance_fifth += resistance.iloc[index5]
            resistance_values_fifth.append(resistance.iloc[index5])
            index5 += 1

        # Calculate the average and SEM for the fifth while loop
        average_fifth = sum_resistance_fifth / len(resistance_values_fifth)

        # Calculate the response for each resistance point after the first while loop
        response = (average_first - resistance) / resistance

        # Calculate main response
        response_main = (average_first - average_third) / average_third

        # Fill in missing points by connecting the two points around it
        response = response.interpolate(method='cubic')

        # Calculate the response time
        response_time = time.iloc[index2] - time.iloc[index]

        # Calculate the recovery time
        recovery_time = time.iloc[index4] - time.iloc[index3]

        # Update the gas_result_window with sensor results
        sensor_label = ttk.Label(result_frame, text=f"{sensor}:")
        sensor_label.pack()

        response_label = ttk.Label(result_frame, text=f"Main Response: {response_main}")
        response_label.pack()

        response_time_label = ttk.Label(result_frame, text=f"Response Time: {0.9 * response_time}")
        response_time_label.pack()

        recovery_time_label = ttk.Label(result_frame, text=f"Recovery Time: {0.9 * recovery_time}")
        recovery_time_label.pack()

        # Turn off interactive mode to prevent plots from being displayed on the console
        plt.ioff()

        # Plot the resistance vs. time and response vs. time
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        sns.lineplot(data=df, x='Time', y=sensor, ax=axes[0])
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Resistance (Ohms)')
        axes[0].set_title(sensor)

        sns.lineplot(data=df, x='Time', y=response, ax=axes[1])
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Response')
        axes[1].set_title(sensor)

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.25)

        # Create a FigureCanvasTkAgg instance
        canvas = FigureCanvasTkAgg(fig, master=result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

        plt.close(fig)

    # Update the scrollbar region after adding all the widgets
    result_frame.update_idletasks()
    canvas.get_tk_widget().configure(scrollregion=canvas.get_tk_widget().bbox("all"))

    # Make the gas_result_window visible
    gas_result_window.deiconify()

# Create a Tkinter window
window = tk.Tk()
window.title("Analyzer for Gas Sensors")

# Create a custom font using the Font class
Title_font = font.Font(family="Arial", size=16, weight="bold")

# Create a custom font using the Font class
Sub_font = font.Font(family="Arial", size=7, weight="normal")

# Create a custom font using the Font class
Norm_font = font.Font(family="Arial", size=11, weight="normal")

# Get the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate the desired width and height for the main frame
desired_width = int(screen_width * 0.5)
desired_height = int(screen_height * 0.5)

# Set the initial size of the main window
window.geometry(f"{desired_width}x{desired_height}")

# Create the main frame to hold all the content
main_frame = ttk.Frame(window)

# Create a new frame for the "Open File" button
PCA_frame = ttk.Frame(window)

# Create a new frame for the "Open File" button
Gas_frame = ttk.Frame(window)

# Create a new tkinter window for displaying PCA results
pca_result_window = tk.Toplevel()
pca_result_window.title("PCA Results")
pca_result_window.withdraw()

# Create a new tkinter window for displaying PCA results
gas_result_window = tk.Toplevel()
gas_result_window.title("Gas Response Results")
gas_result_window.withdraw()

# Create a label for the textbox
Main_label = ttk.Label(main_frame, text="Gas Sensor Analyzer", font=Title_font)
Main_label.pack()

# Create a label for the textbox
Name_label = ttk.Label(main_frame, text="Made by Ishaan Bansal", font=Sub_font)
Name_label.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(main_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
Main_label2 = ttk.Label(main_frame, text="Make sure your data is in an excel file. The format for each file specified in their respective frame.", font=Norm_font)
Main_label2.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(main_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
Main_label3 = ttk.Label(main_frame, text="PCA (Principle Component Analysis) is used for analysis of multiple gases used on exactly 4 sensors.", font=Norm_font)
Main_label3.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(main_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
Main_label4 = ttk.Label(main_frame, text="Gas Response is used for analysis of 1 gas used on exactly 4 sensors.", font=Norm_font)
Main_label4.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(main_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a button labeled "PCA"
PCA_button = ttk.Button(main_frame, text="PCA", command=open_PCA_frame)
PCA_button.pack(side=tk.LEFT, padx=175, pady=10)

# Create a button labeled "Gas"
Gas_button = ttk.Button(main_frame, text="Gas Response", command=open_Gas_frame)
Gas_button.pack(side=tk.LEFT, padx=0, pady=10)

# Create a label for the textbox
MainP_label = ttk.Label(PCA_frame, text="PCA", font=Title_font)
MainP_label.pack()

# Load the image using PIL
image = Image.open(resource_path("C:\\Users\\bansa\\PythonProjects\\UH Gas Sensors\\Gas Analyzer\\images\\PCA_info.png"))
#image = Image.open("C:\\Users\\bansa\\PythonProjects\\UH Gas Sensors\\Gas Analyzer\\dist\\images\\PCA_info.png")

# Resize the image using LANCZOS resampling
desired_image_width = 250
desired_image_height = 150
image = image.resize((desired_image_width, desired_image_height), Image.LANCZOS)

# Create a PhotoImage object from the image
photo_image = ImageTk.PhotoImage(image)

# Create a label to display the image
image_label = ttk.Label(PCA_frame, image=photo_image)
image_label.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(PCA_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
P_label2 = ttk.Label(PCA_frame, text="The 1st 4 columns will be readings for each sensor, the 5th will be gas types, and 6th will be the day.", font=Norm_font)
P_label2.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(PCA_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
P_label3 = ttk.Label(PCA_frame, text="Make sure that there are no columns names, and that the days have the word 'Day' infront of it.", font=Norm_font)
P_label3.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(PCA_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
gases_label = ttk.Label(PCA_frame, text="Enter target gases (separated by commas):")
gases_label.pack()

# Create a textbox for entering the target gases
gases_entry = ttk.Entry(PCA_frame, width=50)
gases_entry.pack()

# Create a label for the textbox
MainG_label = ttk.Label(Gas_frame, text="Gas Response", font=Title_font)
MainG_label.pack()

# Load the image using PIL
#Gimage = Image.open("C:\\Users\\bansa\\PythonProjects\\UH Gas Sensors\\Gas Analyzer\\dist\\images\\GasResponse_info.png")  # Replace this with the actual path to your image file
Gimage = Image.open(resource_path("C:\\Users\\bansa\\PythonProjects\\UH Gas Sensors\\Gas Analyzer\\images\\GasResponse_info.png"))

# Resize the image using LANCZOS resampling
desired_image_width = 250
desired_image_height = 150
Gimage = Gimage.resize((desired_image_width, desired_image_height), Image.LANCZOS)

# Create a PhotoImage object from the image
Gphoto_image = ImageTk.PhotoImage(Gimage)

# Create a label to display the image
Gimage_label = ttk.Label(Gas_frame, image=Gphoto_image)
Gimage_label.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(Gas_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
P_label2 = ttk.Label(Gas_frame, text="The 1st column will be the 'Time' column, and the next 4 columns will be the sensors and their names.", font=Norm_font)
P_label2.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(Gas_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
P_label3 = ttk.Label(Gas_frame, text="As of now the Gas response can only be run once everytime the app is open, but I am working on a fix.", font=Norm_font)
P_label3.pack()

# Create a label for the textbox
Minfo_label = ttk.Label(Gas_frame, text="", font=Norm_font)
Minfo_label.pack()

# Create a label for the textbox
sensors_label = ttk.Label(Gas_frame, text="Enter target sensors (separated by commas):")
sensors_label.pack()

# Create a textbox for entering the target gases
sensors_entry = ttk.Entry(Gas_frame, width=50)
sensors_entry.pack()

# Create the "Open File" button
openP_button = ttk.Button(PCA_frame, text="Open File for PCA", command=perform_PCA_with_user_input)
openP_button.pack()

# Create the "Open File" button
openG_button = ttk.Button(Gas_frame, text="Open File for Gas Response", command=perform_Gas_with_user_input)
openG_button.pack()

#Create back button for PCA
backP_button = ttk.Button(PCA_frame, text="Back", command=goP_Home)
backP_button.pack()

#Create back button for Gas
backG_button = ttk.Button(Gas_frame, text="Back", command=goG_Home)
backG_button.pack()

# Create a label for displaying PCA information
pca_info_label = ttk.Label(pca_result_window, text="PCA Information:")
pca_info_label.pack()

# Create a new tkinter window for displaying Gas results
gas_result_window = tk.Toplevel()
gas_result_window.title("Gas Results")
gas_result_window.withdraw()

# Create a label for displaying Gas information
gas_info_label = ttk.Label(gas_result_window, text="Gas Information:")
gas_info_label.pack()

# Create a canvas for scrolling
canvas = tk.Canvas(gas_result_window)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Create a vertical scrollbar
scrollbar = tk.Scrollbar(gas_result_window, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas and scrollbar
canvas.configure(yscrollcommand=scrollbar.set)

# Function to update the scrollbar region
def update_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
    canvas.configure(yscrollcommand=scrollbar.set)

# Create a frame inside the canvas
result_frame = ttk.Frame(canvas)
# Add the frame to the canvas
canvas.create_window((0, 0), window=result_frame, anchor='nw')

# Bind the frame to the <Configure> event to update the scrollbar region
result_frame.bind("<Configure>", update_scroll_region)

# Allow scrolling with mouse wheel
gas_result_window.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 * int((e.delta / 120)), "units"))

# Calculate the x and y coordinates to position the main frame in the middle of the screen
x = (window.winfo_screenwidth() - desired_width) // 2 - 100  # Adjust the value as needed
y = (window.winfo_screenheight() - desired_height) // 2

# Position the main frame using the place() method
main_frame.pack()

# Run the Tkinter event loop
window.mainloop()


# In[ ]:




