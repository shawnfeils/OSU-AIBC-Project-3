# OSU-AIBC-Project-3
## Overview
For our final project, we focused on developing a handwriting recognition application utilizing neural networks. Our primary goal was to train a model capable of accurately recognizing handwritten digits and letters. We began by familiarizing ourselves with model training and image processing techniques, leveraging the MNIST and EMNIST datasets available through TensorFlow.

## Approach
### Model Development:
- We started with the MNIST dataset, aiming to build a model that could accurately predict handwritten digits. After some iterations, we successfully trained the model to achieve satisfactory accuracy.
- **Incorporating Letters**: Next, we transitioned to the EMNIST dataset, which includes both letters and digits. We initially attempted to train the model using the EMNIST letters dataset but encountered challenges in achieving the desired accuracy.
- **Switching to EMNIST by Class**: We switched to the EMNIST by class dataset, which contains both letters and numbers. Despite our expectations, this approach proved more complex than anticipated:
  * We performed hyperparameter tuning, increased the number of epochs, and applied data augmentation to improve model performance.
  * We attempted to utilize the Keras Tuner for optimizing hyperparameters, but the process was time-consuming and ultimately abandoned after running for over 25 hours without completion.
- **Utilizing External Datasets**: Following feedback from Yuyang, we sourced a high-performing dataset from Kaggle. We imported this dataset as a CSV and adopted the original author’s image preprocessing steps while maintaining our prediction code from the digit model.
  * Despite achieving high accuracy with the new dataset, the model struggled with accurate predictions, such as distinguishing between serif and sans serif capital “E” which it returned as an "A" and "D" respectively with 100% certainty





For our final project, we decided to focus on handwriting recognition application using a Neural Network. The first step was to get comfortable with training a model and processing images

Both MNIST and EMNIST data sets were loaded from tensor flow.

Started simple - can we build an MNIST model, train it and accurately predict new images and our hand written numbers. It took some time to get there but it got there. 

We then did Extended MNIST letters to incorporate letters into our training model.  EMNIST letter dataset from

Switched to EMNIST by class has both letters and numbers; data is preloaded in tensorflow
- thought it would be just as easy as the digits and it was not
- it did not work
- we did hyperparamenter tuning, additional epochs, did augmentation of the training dataset, attempted to run the keras tuner to try to get to a more accurate model because the model was in the 83-86% accuracy range 
- tried to run the keras tuner which runs the models in series; randomizes the model parameters and runs the model repeatedly to optimize the hyperparameter settings - this ran for over 25 hours and did not complete; therefore we abandoned it
- took yuyang's feedback and grabbed a high performing dataset from Kaggle and that is where we used the code base there and imported the data set as a CSV that was referenced, and we followed their lead on the training data image pre-processing foor the data set; used the oirginal author's image pre-processing steps for the training data set, but continued to use our prediction code we used for the original digits; the accuracy is through the roof on that one so it is highly accurate - and it did not make a good prediction
- gave it two capital E's - one that was serif (A) and one sans serif (D) so it did not predict correctly - and thought it was 100% right 
- conclusion: highly trained model not making good predictions
- list of things we can continue to do to try to double check the predictions



Our story - we got this thing going well with digits, but it did not translate well into the handwriting dataset and it was not successful. We tried many different approaches and  it did not work. Errors may be in the classification. 






## Approach
1. 
2. 



### Data Collection

### Data Pre-processing

### Model Building

### Training

### Evaluation

### Prediction

### Conclusion










Model Implementation (25 points)
There is a Jupyter notebook that thoroughly describes the data extraction, cleaning, preprocessing, and transformation process, and the cleaned data is exported as CSV files for a machine or deep learning model, or natural language processing (NLP) application. (10 points) Convolutional Nueral Network, and a simple neural network

A Python script initializes, trains, and evaluates a model or loads a pretrained model. (10 points) Correct

At least one additional library or technology NOT covered in class is used. (5 points) MNIST & Extended MNIST & Keras Tuner

Model Optimization (25 points)
The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is documented in either a CSV file, Excel table, or in the Python script itself. (15 points) Documented in the python script

Overall model performance is printed or displayed at the end of the script. (10 points) Correct

GitHub Documentation (25 points)
The GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use. (10 points) yes

The README is customized as a polished presentation of the content of the project. (15 points) yes

Presentation Requirements (25 points)
Your presentation should cover the following:

An executive summary or overview of the project and project goals. (5 points)

An overview of the data collection, cleanup, and exploration processes. Include a description of how you evaluated the trained model(s) using testing data. (5 points)

The approach that your group took to achieve the project goals. (5 points)

Any additional questions that surfaced, what your group might research next if more time was available, or a plan for future development. (3 points)

The results and conclusions of the application or analysis. (3 points)

Slides that effectively demonstrate the project. (2 points)

Slides that are visually clean and professional. (2 points)


###
Trolly notes - 