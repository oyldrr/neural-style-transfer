# Stilist: Neural-Artistic Style Transfer Web Application

## Overview
**Stilist** is a web application project leveraging Neural-Artistic Style Transfer technology. By using deep learning architectures, the project generates a new image that blends the content of one image with the style of another. The application provides a user-friendly and intuitive interface to simplify the process.

## Application Screenshots
![Ekran görüntüsü 2024-12-06 174420](https://github.com/user-attachments/assets/2bc5c801-4617-4282-86bb-d01fc36aed91)
![Ekran görüntüsü 2024-12-06 174234](https://github.com/user-attachments/assets/43ada4c4-b246-42a6-b798-c2375c70e0d0)
![Ekran görüntüsü 2024-12-06 174058](https://github.com/user-attachments/assets/2b9d042b-8241-4f4b-b7b4-abcaf86ae86c)


## What is Neural-Artistic Style Transfer (NST)?
Neural-Artistic Style Transfer involves recreating a content image while preserving its structural details, using the stylistic features of another image. This is achieved through deep neural networks.

### Key Concepts:
- **Content Image**: Typically landscapes or any user-provided image.
- **Style Image**: Images emphasizing patterns, textures, and colors, often derived from surrealist art.

## Applications of NST
Neural-Artistic Style Transfer technology has applications across various industries to enhance creativity and generate visually striking results. 

### Common Use Cases:
1. **Art & Digital Art**: Creating artistic visual projects.
2. **Advertising & Marketing**: Generating eye-catching content.
3. **Gaming & Entertainment**: Designing aesthetic game worlds and animations.
4. **Fashion & Textile**: Producing unique clothing and textile designs.
5. **Medical Imaging**: Visualizing medical data innovatively.
6. **Education & Research**: Teaching AI and computer vision concepts.
7. **Interior & Architectural Design**: Crafting unique interior spaces.
8. **Cultural & Historical Preservation**: Digitally restoring and displaying artifacts.
9. **Personalized Gifts**: Creating custom gifts and decorative products.

NST expands the boundaries of creativity, offering impactful and aesthetic solutions in both art and technology.

## Neural-Artistic Style Transfer Model

### Libraries Used:
This project uses key libraries such as TensorFlow, NumPy, and Matplotlib. The pre-trained **VGG19** model plays a central role in the NST process.

### About VGG19:
VGG19 is a deep neural network architecture developed by the Visual Geometry Group (VGG) at Oxford University. It achieved notable success in the 2014 ImageNet Large Scale Visual Recognition Challenge (ILSVRC).

#### Key Features of VGG19:
- **Depth**: 19 layers (16 convolutional + 3 fully connected).
- **Convolutional Layers**: Uses fixed 3x3 filters to capture intricate features.
- **Activation Function**: ReLU for non-linearity.
- **Pooling**: Max pooling for feature map reduction.
- **Fully Connected Layers**: Utilized for classification tasks.

### Workflow:
1. **Image Preprocessing**: Input images are preprocessed for the model and then reverted for display.
2. **Model Loading**: The VGG19 model is loaded with parameters adjusted for feature extraction.
3. **Feature Extraction**: Features from the content and style images are extracted using specific layers of VGG19.
4. **Loss Computation**:
   - **Content Loss**: Ensures the structural integrity of the content image.
   - **Style Loss**: Maintains the textural and color properties of the style image.
   - **Total Loss**: A weighted combination of content and style losses.
5. **Optimization**: The generated image is iteratively optimized to minimize total loss.

## Application Workflow
1. **Input Images**: Users upload content and style images.
2. **Processing**: NST algorithm generates the blended image.
3. **Results**: The output is displayed and available for download.

## Screenshots
- **Home Screen**
- **Loading Screen**
- **Result Screen**

## References
1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). A Neural Algorithm of Artistic Style. *arXiv preprint* [arXiv:1508.06576](https://arxiv.org/abs/1508.06576).
2. TensorFlow. (n.d.). Style Transfer. Retrieved December 6, 2024, from [TensorFlow Tutorials](https://www.tensorflow.org/tutorials/generative/style_transfer?hl=en).
