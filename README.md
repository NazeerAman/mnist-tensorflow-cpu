This repository contains code for building a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset.

**Functionality**
Downloads the MNIST dataset.
Preprocesses the data by normalizing pixel values.
Defines a CNN architecture with convolutional and dense layers.
Trains the model on the training data.
Evaluates the model's performance on the testing data.
Visualizes the training and validation loss and accuracy.
Analyzes the effect of batch size on training accuracy.
Predicts the class labels for new images.
Generates confusion matrices to visualize model performance.


**Running the Code**
This code requires the following libraries:
tensorflow
numpy
pandas
seaborn
matplotlib.pyplot
cv2 (OpenCV)
Clone this repository.
Install the required libraries using pip install <library_name>.

**Code Structure**
MNIST.ipynb: Jupyter notebook containing the complete code for training, evaluation, and prediction.

**Output**
The script will generate the following outputs:
Images of sample training data with their corresponding labels.
Plots of training and validation loss and accuracy.
Confusion matrices for both the training and testing datasets.
Predicted labels for new input images.

**Additional Notes**
This code uses a basic CNN architecture. More complex architectures can be explored for further improvement.
The script provides functionalities for analyzing batch size effects and evaluating model performance.
You can modify the script to include your own image pre-processing and prediction functionalities.
