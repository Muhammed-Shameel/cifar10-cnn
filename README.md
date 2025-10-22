CIFAR-10 Image Classification Using CNN

This project builds and trains a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 categories, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

📁 Project Structure
cifar10-cnn/
│
├── data/                # Dataset loaded automatically from Keras
├── notebooks/           # Jupyter notebook version of training
├── results/             # Accuracy plots and confusion matrix
├── README.md            # Project description
└── requirements.txt     # Dependencies

🚀 Overview

The project demonstrates:

Building a CNN model using TensorFlow/Keras

Training on CIFAR-10, a dataset of 60,000 32×32 color images

Evaluating model accuracy and loss

Testing generalization using validation data

Saving and reusing trained models

📊 Model Performance
Metric	Result
Training Accuracy	71.5%
Validation Accuracy	74.6%
Training Loss	0.976
Validation Loss	0.883
Epochs	50
Learning Rate	0.0002

⚠️ Note: CIFAR-10 images are only 32×32 pixels, which makes it difficult for models to distinguish similar objects like cats and dogs. The relatively low resolution limits the achievable accuracy for simple CNNs.

🧩 Model Architecture

The CNN model consists of multiple convolution and pooling layers followed by dense layers.
It is designed to balance accuracy and computational efficiency.

Input (32×32×3)
↓
Conv2D(96 filters, 3×3, activation='relu')
↓
MaxPooling2D(2×2)
↓
Dropout(0.25)
↓
Conv2D(96 filters, 3×3, activation='relu')
↓
MaxPooling2D(2×2)
↓
Dropout(0.25)
↓
Flatten
↓
Dense(192, activation='relu')
↓
Dropout(0.5)
↓
Dense(10, activation='softmax')


Total Parameters: 1,272,106
Optimizer: Adam
Loss Function: Categorical Crossentropy
Learning Rate: 0.0002
Epochs: 50
Final Validation Accuracy: ~74.6%
⚙️ Installation

Clone the repository:

git clone https://github.com/yourusername/cifar10-cnn.git
cd cifar10-cnn


Install dependencies:

pip install -r requirements.txt


Run training:

python src/train_cifar10.py

📈 Results Visualization

Training & validation accuracy/loss plots

Confusion matrix for misclassified samples

Random test predictions visualized with actual labels

(All stored in the results/ folder)

💡 Future Improvements

✅ Image Augmentation (rotation, zoom, flips)

✅ Batch Normalization for stable training

✅ Transfer Learning using pretrained networks (MobileNetV2 / ResNet50)

🔜 CIFAR-100 dataset for higher challenge

🔜 Web app deployment (Streamlit / Flask demo)

🧾 Dataset

Source: CIFAR-10 (Keras built-in)

Size: 60,000 images (50,000 training + 10,000 testing)

Image Dimensions: 32×32×3

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

📚 Tech Stack

Language: Python

Libraries: TensorFlow, Keras, NumPy, Matplotlib

IDE: VS Code

🏁 Author

Muhammed Shameel
📧 muhammedshameel3009@email.com
