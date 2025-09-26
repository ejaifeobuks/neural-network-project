# Neural Network for Handwritten Digit Classification

This is a machine learning project that implements a neural network to classify handwritten digits (0-9) using the MNIST dataset with TensorFlow/Keras. It is a classical beginner project to get understanding of neural networks

## 📋 Project Overview

This project demonstrates the implementation of a feedforward neural network for image classification. The model is trained on the famous MNIST dataset, which contains 70,000 images of handwritten digits, and achieves high accuracy in digit recognition.

## 🧠 Model Architecture

The neural network consists of:

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

## 📊 Dataset

- **Source**: MNIST Handwritten Digit Database
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)

## 🛠️ Technologies Used

- **Python 3.11+**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning utilities
- **Matplotlib** - Data visualization (if used)
- **Jupyter Notebooks** - Interactive development environment

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.11+ installed. You can check your Python version with:

```bash
python --version
```

### Installation

1. **Clone the repository** (if applicable):

   ```bash
   git clone <repository-url>
   cd neural-network-digit-classification
   ```

2. **Create a virtual environment**:

   ```bash
   # Using conda (recommended)
   conda create -n tensorflow-env python=3.11 -y
   conda activate tensorflow-env

   # Or using venv
   python -m venv neural-net-env
   source neural-net-env/bin/activate  # Linux/Mac
   # neural-net-env\Scripts\activate  # Windows
   ```

3. **Install required packages**:

   ```bash
   pip install tensorflow scikit-learn numpy matplotlib jupyter ipykernel
   ```

4. **Launch Jupyter Notebook**:

   ```bash
   jupyter notebook Neural_Network_Project.ipynb
   ```

   Or open in VS Code:

   ```bash
   code Neural_Network_Project.ipynb
   ```

## 📖 Usage

### Running the Project

1. **Open the notebook**: `Neural_Network_Project.ipynb`

2. **Run cells sequentially**:

   - **Cell 1**: Import required libraries
   - **Cell 2**: Load MNIST dataset
   - **Cell 3**: Normalize pixel values (0-1 range)
   - **Cell 4**: Reshape images to 1D arrays
   - **Cell 5**: Convert labels to one-hot encoding
   - **Cell 6**: Build, train, and evaluate the model

3. **View results**: The model will display training progress, test accuracy, and detailed classification metrics.

### Expected Output

- **Training**: ~10 epochs with validation accuracy reaching 95%+
- **Test Accuracy**: Typically achieves 97-98% accuracy
- **Classification Report**: Precision, recall, and F1-score for each digit class

## 📈 Model Performance

The neural network typically achieves:

- **Training Accuracy**: ~98%
- **Validation Accuracy**: ~97%
- **Test Accuracy**: ~97-98%

### Training Process

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Epochs**: 10
- **Batch Size**: 32
- **Validation Split**: 20%

## 🔍 Key Features

- **Data Preprocessing**: Automatic normalization and reshaping
- **One-Hot Encoding**: Proper categorical label encoding
- **Model Validation**: Built-in validation split during training
- **Performance Metrics**: Comprehensive evaluation with classification report
- **Efficient Architecture**: Balanced network size for optimal performance

## 📁 Project Structure

```
neural-network-digit-classification/
│
├── Neural_Network_Project.ipynb    # Main notebook with complete implementation
├── README.md                       # Project documentation (this file)
└── requirements.txt               # Python dependencies (if created)
```

## 🎓 Learning Outcomes

This project demonstrates:

- **Neural Network Fundamentals**: Architecture design and implementation
- **Data Preprocessing**: Image normalization and formatting
- **Deep Learning Workflow**: Train-validation-test pipeline
- **Model Evaluation**: Understanding accuracy, precision, recall metrics
- **TensorFlow/Keras Usage**: Practical deep learning framework application

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements:

- Additional data visualization
- Model architecture experiments
- Performance optimization
- Documentation enhancements

## 📄 License

This project is for educational purposes. The MNIST dataset is publicly available and free to use.

## 📧 Contact

**Author**: Ogheneobukome Ejaife
**Course**: CS 215-80, Fall 2025

---

## 🚀 Quick Start Commands

```bash
# 1. Create environment
conda create -n tensorflow-env python=3.11 -y && conda activate tensorflow-env

# 2. Install dependencies
pip install tensorflow scikit-learn numpy matplotlib jupyter ipykernel

# 3. Run notebook
code Neural_Network_Project.ipynb

# 4. Select tensorflow-env kernel in VS Code and run all cells!
```

**Happy Learning! 🎉**
