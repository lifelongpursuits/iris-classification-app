import os
import sys
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import IrisClassificationModel

# Add comprehensive path debugging
print("DEBUG: Python Executable Path:", sys.executable)
print("DEBUG: Current Working Directory:", os.getcwd())
print("DEBUG: Script Location:", os.path.abspath(__file__))
print("DEBUG: Python Path:", sys.path)

def get_species_images(species):
    """
    Find and return image paths for a given species
    Assumes images are stored in the 'images' directory with filenames like:
    setosa_1.jpg, setosa_2.jpg, versicolor_1.jpg, etc.
    """
    # Try multiple potential image directory locations
    potential_dirs = [
        r'C:\Users\Name\.vscode\Repos\iris-classification-app\images',  # Absolute path
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images'),  # Relative to script
        os.path.join(os.getcwd(), 'images'),  # Current working directory
        'images'  # Fallback to simple 'images' directory
    ]
    
    # Try each potential directory
    for image_dir in potential_dirs:
        print(f"DEBUG: Attempting to find images in: {image_dir}")
        
        # Check if directory exists
        if not os.path.exists(image_dir):
            print(f"DEBUG: Directory does not exist: {image_dir}")
            continue
        
        # List all files in the directory
        try:
            all_files = os.listdir(image_dir)
            print(f"DEBUG: Files in {image_dir}: {all_files}")
        except Exception as e:
            print(f"ERROR: Could not list directory contents: {e}")
            continue
        
        # Filter images for the specific species
        species_images = [
            os.path.join(image_dir, f) 
            for f in all_files 
            if f.lower().startswith(species.lower()) and f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # If we found images, return them
        if species_images:
            print(f"DEBUG: Found images for {species}: {species_images}")
            return species_images
    
    # If no images found in any location
    print(f"ERROR: No images found for species {species}")
    return []

def plot_flower_measurements(sepal_length, sepal_width, petal_length, petal_width):
    """Create a radar chart of flower measurements"""
    categories = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    values = [sepal_length, sepal_width, petal_length, petal_width]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    values += values[:1]
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Flower Measurements')
    
    return fig

def get_species_info(species):
    """Return detailed information about the predicted Iris species"""
    species_info = {
        'setosa': {
            'description': 'Setosa is the most distinct of the three Iris species. It has the smallest petals and is easily distinguishable.',
            'characteristics': ['Smallest petals', 'Compact flower structure', 'Typically found in cool, moist environments'],
            'wiki_link': 'https://en.wikipedia.org/wiki/Iris_setosa'
        },
        'versicolor': {
            'description': 'Versicolor is a medium-sized Iris with blue-violet to lavender-blue flowers.',
            'characteristics': ['Medium-sized petals', 'Blue-violet coloration', 'Common in meadows and open woodlands'],
            'wiki_link': 'https://en.wikipedia.org/wiki/Iris_versicolor'
        },
        'virginica': {
            'description': 'Virginica is the largest of the three Iris species, with large, showy flowers.',
            'characteristics': ['Largest petals', 'Deep purple or blue-purple flowers', 'Found in wet areas and along waterways'],
            'wiki_link': 'https://en.wikipedia.org/wiki/Iris_virginica'
        }
    }
    return species_info.get(species.lower(), {'description': 'Unknown species', 'characteristics': [], 'wiki_link': '#'})

def main():
    st.title('Iris Flower Species Classifier ')
    
    # Sidebar for input
    st.sidebar.header('Enter Flower Measurements')
    
    # Input sliders for features
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.0, 0.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.0, 0.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 4.0, 0.1)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 1.0, 0.1)
    
    # Prepare features for prediction
    features = [sepal_length, sepal_width, petal_length, petal_width]
    
    # Load pre-trained model
    try:
        model = IrisClassificationModel.load_model()
    except FileNotFoundError:
        st.error("Model not found. Please train the model first by running model.py")
        return
    
    # Prediction button
    if st.sidebar.button('Predict Species'):
        # Make prediction
        prediction = model.predict(features)
        
        # Display prediction
        st.success(f'Predicted Iris Species: {prediction}')
        
        # Visualization and additional info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Flower Measurement Details')
            st.write(f"Sepal Length: {sepal_length} cm")
            st.write(f"Sepal Width: {sepal_width} cm")
            st.write(f"Petal Length: {petal_length} cm")
            st.write(f"Petal Width: {petal_width} cm")
            
            # Radar chart of measurements
            st.pyplot(plot_flower_measurements(sepal_length, sepal_width, petal_length, petal_width))
        
        with col2:
            st.subheader('Species Information')
            species_info = get_species_info(prediction)
            st.write(species_info['description'])
            st.markdown('**Key Characteristics:**')
            for char in species_info['characteristics']:
                st.markdown(f'- {char}')
            
            # Add Wikipedia link
            st.markdown(f"[Learn more about {prediction.capitalize()} Iris on Wikipedia]({species_info['wiki_link']})")
            
            # Display species images
            st.subheader('Example Images')
            species_images = get_species_images(prediction)
            
            if species_images:
                # Display the first image
                st.image(species_images[0], use_container_width=True)
            else:
                st.warning('No images found for this species')
    
    # About section
    st.sidebar.markdown('### About the App')
    st.sidebar.info(
        'This app uses a K-Nearest Neighbors Classifier to predict '
        'Iris flower species based on sepal and petal measurements. '
        'Adjust the sliders and click "Predict Species" to classify a flower!'
    )
    
    # Dataset overview
    st.sidebar.markdown('### Dataset Overview')
    st.sidebar.write('The Iris dataset contains 3 species:')
    st.sidebar.write('- Iris Setosa')
    st.sidebar.write('- Iris Versicolor')
    st.sidebar.write('- Iris Virginica')

if __name__ == '__main__':
    main()
