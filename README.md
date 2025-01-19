# MNIST Neural Network Trainer

A graphical user interface (GUI) application written in Rust for training and testing neural networks on the MNIST dataset. Customize your neural network settings, train the model, evaluate its performance, and make predictions with ease.

## Table of Contents
- [Features](#Features)
- [Installation](#Installation)
- [Usage](#Usage)
- [Configuration](#Configuration)
- [Screenshots](#Screenshots)
- [Contributing](#Contributing)
- [License](#License)

## Features
- **Customizable Network Architecture**: Define the number of layers, neurons per layer, and activation functions.
- **Training Controls**: Start, pause, resume, and stop the training process.
- **Real-Time Metrics**: Monitor training progress, training accuracy, and testing accuracy.
- **Model Persistence**: Save and load trained models for future use.
- **Prediction Interface**: Select test samples to make predictions and compare with actual labels.
- **Logging**: View real-time status updates and logs within the application.

## Installation
### Prerequisites
- Operating System: Windows
- Rust: Ensure you have Rust installed if you plan to build from source.

## Download Executable
1. Visit the [GitHub Releases](https://github.com/morgan-mcnabb/mnist-trainer/releases/tag/v0.1.0) page.
2. Download the latest trainer-0.1.exe.
3. Create a folder where you want to store the executable.
4. Inside this folder, create a subfolder named data.
5. Download the MNIST dataset from [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) and place the dataset files (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte) inside the data folder.
```
your-folder/
├── trainer-0.1.exe
└── data/
    ├── train-images-idx3-ubyte
    ├── train-labels-idx1-ubyte
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

## Build from Source (Optional)
If you prefer to build the application from source:

1. Clone the repository:
  ```git clone https://github.com/yourusername/mnist-neural-network-trainer.git```
2. Navigate to the project directory:
   ```cd mnist-neural-network-trainer```
3. Build the project:
  ```cargo build --release```
4. The executable will be located in target/release/.

## Usage
1. Ensure the data folder with the MNIST dataset is in the same directory as the executable.
2. Run the executable:
    - Windows: Double-click trainer-0.1.exe or run it via the command prompt.
3. The GUI will launch, presenting various configuration options and controls.

## Configuration
### Network Settings
- **Epochs**: Number of times the entire training dataset passes through the network. Default is ```20```.
- **Learning Rate**: The step size for updating weights during training. Default is ```0.1```.
- **Layers**: Define the size of each layer in the network. Enter comma-separated values (e.g., ```784,128,64,10```).
- **Activations**: Specify the activation functions for each layer (excluding the input layer). Enter comma-separated values (e.g., sigmoid,relu,relu).
  
### Training Controls
- **Start Training**: Begin the training process with the current configuration.
- **Pause Training**: Temporarily halt the training process.
- **Resume Training**: Continue training from where it was paused.
- **Stop Training**: Terminate the training process.
  
### Model Management
- **Save Model**: Save the current trained model to `trained_model.json`.
- **Load Model**: Load a previously saved model from `trained_model.json`.
 
### Prediction
- **Select Test Sample Index**: Choose an index to select a test sample for prediction.
- **Predict**: Run the model to predict the label of the selected test sample. The predicted label and actual label will be displayed.

### Screenshots
_Screenshots and gifs coming later!_

### Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Push to your fork and submit a pull request.
   
For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License.
