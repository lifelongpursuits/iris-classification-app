import base64
import os

# Hardcoded base64 encoded images for each Iris species
iris_images = {
    'setosa': [
        # A sample base64 encoded image placeholder
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==',
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='
    ],
    'versicolor': [
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==',
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='
    ],
    'virginica': [
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg==',
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=='
    ]
}

# Ensure images directory exists
os.makedirs('images', exist_ok=True)

# Save base64 images
for species, images in iris_images.items():
    for i, img_base64 in enumerate(images, 1):
        filename = f'images/{species}_{i}.jpg'
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_base64))
        print(f'Created placeholder image: {filename}')

print('Image generation complete!')
