# PetFinder.my-Adoption-Prediction-Machine-Learning-Project

## Table of Contents

* [About the Project](#about-the-project)
* [Project Goal](#project-goal)
* [Dataset](#dataset)
* [Methodology](#methodology)
* [Results](#results)

## About the Project

This project explores the use of machine learning to predict pet adoption rates on PetFinder.my. 

## Project Goal

The goal of this project is to develop a machine learning model that can predict whether a pet listed on PetFinder.my will be adopted or not. This can be valuable for animal shelters and rescue organizations to optimize their adoption strategies.

## Dataset
* Source of the data (https://www.kaggle.com/c/petfinder-adoption-prediction/data)

**Data Fields:**
- PetID - Unique hash ID of pet profile
- AdoptionSpeed - Categorical speed of adoption. Lower is faster. This is the value to predict. See below section for more info.
- Type - Type of animal (1 = Dog, 2 = Cat)
- Name - Name of pet (Empty if not named)
- Age - Age of pet when listed, in months
- Breed1 - Primary breed of pet (Refer to BreedLabels dictionary)
- Breed2 - Secondary breed of pet, if pet is of mixed breed (Refer to BreedLabels dictionary)
- Gender - Gender of pet (1 = Male, 2 = Female, 3 = Mixed, if profile represents group of pets)
- Color1 - Color 1 of pet (Refer to ColorLabels dictionary)
- Color2 - Color 2 of pet (Refer to ColorLabels dictionary)
- Color3 - Color 3 of pet (Refer to ColorLabels dictionary)
- MaturitySize - Size at maturity (1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large, 0 = Not Specified)
- FurLength - Fur length (1 = Short, 2 = Medium, 3 = Long, 0 = Not Specified)
- Vaccinated - Pet has been vaccinated (1 = Yes, 2 = No, 3 = Not Sure)
- Dewormed - Pet has been dewormed (1 = Yes, 2 = No, 3 = Not Sure)
- Sterilized - Pet has been spayed / neutered (1 = Yes, 2 = No, 3 = Not Sure)
- Health - Health Condition (1 = Healthy, 2 = Minor Injury, 3 = Serious Injury, 0 = Not Specified)
- Quantity - Number of pets represented in profile
- Fee - Adoption fee (0 = Free)
- State - State location in Malaysia (Refer to StateLabels dictionary)
- RescuerID - Unique hash ID of rescuer
- VideoAmt - Total uploaded videos for this pet
- PhotoAmt - Total uploaded photos for this pet
- Description - Profile write-up for this pet. The primary language used is English, with some in Malay or Chinese.

* Pre-processing steps taken:
- Basic data cleaning such as renaming feature columns, mapping code IDs, handling missing values and dropping features/ data points where applicable
- Data Engineering new features such as binning (discretization) to transform some features with a very high number of unique values into categorical features with fewer categories. Also engineered some other boolean features based on whether pet has name etc.
- Preprocessing for model ingestion: Normalization for numerical features and One hot encoding for categorical features.

## Methodology

**1. Data Preprocessing**
* **Feature Selection:** The code utilizes a pre-defined list of `selected_features` to focus on relevant data points for adoption prediction. Features not included in this list are dropped using `dataCleaningDropFeatures`.
* **Feature Engineering:**
    * **Categorical Features:** Features with string values (e.g., breed) are considered categorical. The code employs `get_category_encoding_layer` to transform these features. This layer learns the unique values present in the training data and assigns them integer indices. Subsequently, a `CategoryEncoding` layer is applied for one-hot encoding, which creates separate binary features for each unique category.
    * **Numerical Features:** Features with numerical values (e.g., age) are considered numerical. The code utilizes `get_normalization_layer` to normalize these features. Normalization ensures all features have a similar scale, improving model training stability.

**2. Model Building**
* **Input Layers:** The code creates separate input layers for both processed categorical and numerical features using `tf.keras.Input`. 
* **Feature Processing Layers:**
    * **Categorical Features:** The encoded categorical features are piped through the layers returned by `get_category_encoding_layer`. 
    * **Numerical Features:** The numerical features are passed through the normalization layers created by `get_normalization_layer`.
* **Concatenation:** After processing, all features (both categorical and numerical) are concatenated using `tf.keras.layers.concatenate` to create a single input for the main model architecture.
* **Hidden Layers:** The core model architecture consists of several densely connected hidden layers. Each layer uses a ReLU (Rectified Linear Unit) activation function for non-linearity and Dropout layers for regularization to prevent overfitting. The number of neurons in each layer (128, 64, 32) can be further tuned for optimal performance.
* **Output Layer:** The final layer has a single neuron with a sigmoid activation function. Sigmoid is suitable for binary classification problems like predicting adoption (adopted or not adopted) as it outputs a probability between 0 and 1.

**3. Model Training**
* **Compilation:** The model is compiled using the Adam optimizer, binary crossentropy loss function, and various evaluation metrics:
    * Accuracy: Measures the proportion of correct predictions.
    * Precision: Measures the ratio of true positives (correctly predicted adopted pets) to all predicted adopted pets.
    * Recall: Measures the ratio of true positives (correctly predicted adopted pets) to all actual adopted pets in the data.
    * AUC (Area Under the ROC Curve): Measures the model's ability to distinguish between adopted and non-adopted pets.
* **Early Stopping:** An EarlyStopping callback is used to monitor validation loss. Training stops if the validation loss fails to improve for a certain number of epochs (patience set to 5), preventing overfitting and saving the best performing model based on validation data.


## Results
Test Loss: 0.5169
Test Accuracy: 0.7430
Test Precision: 0.7762
Test Recall: 0.9170
Test AUC: 0.7148
