# OSU-AIBC-Project-3
## Overview
For our final project, we focused on developing a handwriting recognition application utilizing neural networks. Our primary goal was to train a model capable of accurately recognizing handwritten digits and letters. We began by familiarizing ourselves with model training and image processing techniques, leveraging the MNIST and EMNIST datasets available through TensorFlow. We experienced success with the MNIST dataset and digits, but ultimately could not duplicate those results when transitioning to digits and letters using EMNIST datasets.

## Approach
### Model Development:
- **Digit Recognition Modeling**: We started with the MNIST dataset, aiming to build a model that could accurately predict handwritten digits. After some iterations, we successfully trained the model to achieve satisfactory accuracy.
- **Incorporating Letters**: Next, we transitioned to the EMNIST dataset, which includes both letters and digits. We initially attempted to train the model using the EMNIST letters dataset but encountered challenges in achieving the desired accuracy.
- **Switching to EMNIST by Class**: We switched to the EMNIST by class dataset, which contains both letters and numbers. Despite our expectations, this approach proved more complex than anticipated.
- **Troubleshooting**: We attempted to troubleshoot down many avenues without success:
  * Generate a Confusion Matrix
  * Visualize Misclassified Images
  * Adjust Model Complexity
  * Experiment with Preprocessing
  * Experiment with CNN Model Architecture and Hyperparameter Choices
  - **Utilizing External Datasets**: Following feedback from Yuyang, we sourced a high-performing dataset from Kaggle. We imported this dataset as a CSV and adopted the original author’s image preprocessing steps while maintaining our prediction code from the digit model.
  * Despite achieving high accuracy with the new dataset, the model struggled with accurate predictions, such as distinguishing between serif and sans serif capital “E” which it returned as an "A" and "D" respectively with 100% certainty

## Future Considerations
There are a few considerations that we can take into account to troubleshoot and adjust this model to accurately predict handwritten letters in the future:
* Expand evaluation metrics beyond accuracy to include F1-score, precision, and recall for a more comprehensive assessment of model performance.
* Review the labels and classes to ensure consistency
Future development would include: 
* Incorporating the team’s handwriting into the training data of a successful model
* Incorporating an OCR app to convert long form handwriting into processable images

## Conclusion
Our journey revealed that while we successfully trained a model for digit recognition, the transition to handwriting recognition was fraught with challenges. The highly trained model did not yield reliable predictions for handwritten letters, indicating many possibilities to troubleshoot in the future, such as:
* **Class Imbalance or Incorrect Labels**: The EMNIST ByClass dataset contains 62 classes (letters and digits), which means class imbalance is possible. 
* **Misaligned Input Data**: Work to ensure the proper preprocessing of images to address misalignments, such as centering and data augmentation.
* **Overfitting**: Audit the gap between training and validation accuracy to identify overfitting, and apply regularization techniques like Dropout or Early Stopping to improve generalization.
* **Confusion Between Similar Classes**: Build a confusion matrix to identify misclassified classes and consider focused training or adding more examples for visually similar characters.

## Grading Criteria
Despite a failed result with the letter recognition portion of the project, we were able to successfully predict handwritten digits with our efforts. We were also able to complete all aspects of the project in relation to grading criteria:
* Model Implementation (25 points)
  * There is a Jupyter notebook that thoroughly describes the data extraction, cleaning, preprocessing, and transformation process, and the cleaned data is exported as CSV files for a Convolutional Neural Network and a Simple Neural Network
  * Our Python script initializes, trains, and evaluates a model or loads a pretrained model.
  * We utilized MNIST, EMNIST, and Keras Tuner which were not covered in class
* Model Optimization (25 points)
  * The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is in the Python script itself.
  * Overall model performance is printed or displayed at the end of the script.
* GitHub Documentation (25 points)
  * The GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use
  * The README is customized as a polished presentation of the content of the project.
* Presentation Requirements (25 points)
  * Our presentation should covered the following:
    * An executive summary or overview of the project and project goals.
    * An overview of the data collection, cleanup, and exploration processes. Include a description of how you evaluated the trained model(s) using testing data.
    * The approach that your group took to achieve the project goals.
    * Any additional questions that surfaced, what your group might research next if more time was available, or a plan for future developmement.
    * The results and conclusions of the application or analysis.
    * Slides that effectively demonstrate the project.
    * Slides that are visually clean and professional.