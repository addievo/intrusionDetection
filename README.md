# Intrusion Detection System

- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Output](#output)
  - [model.py](#modelpy)
  - [cli.py](#clipy)
- [About the Author](#about-the-author)
- [License](#license)

This project implements an Intrusion Detection System (IDS) using machine learning techniques. It aims to detect and classify network intrusions or anomalous activities in a computer network.

## Project Description

The IDS is designed to work with training data rather than live data from the system NIC. It utilizes the KDDCup'99 dataset, which is a widely used benchmark dataset for evaluating intrusion detection systems. 

The system consists of two main components: the command-line interface (CLI) and the model.

The CLI allows users to perform various actions such as loading the data, training the model, saving and loading the model, making predictions, and displaying graphs. 

It provides a convenient way to interact with the IDS and perform different tasks based on the user's requirements.

The model component uses a Random Forest Classifier algorithm for training and predicting network intrusions.

It preprocesses the data by encoding categorical features, applying imputation for missing values, and scaling the features. The trained model can then be used to make predictions on new test data.

## Project Structure

The project is organized as follows:

- `cli.py`: Contains the implementation of the CLI, which handles data loading, model training, model saving/loading, prediction, and graph visualization.
- `model.py`: Implements the machine learning model using the Random Forest Classifier algorithm. It handles data preprocessing, model training, and model evaluation.

## Dependencies

The following dependencies are required to run the project:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the dependencies using the following command:

```shell
pip install pandas numpy scikit-learn matplotlib seaborn
```
## Usage

To use the Intrusion Detection System CLI, follow these steps:

1. Install the required dependencies mentioned above.

2. Place the `cli.py` and `model.py` files in the same directory.

3. Open a terminal or command prompt and navigate to the directory where the files are located.

4. Run the following command to execute the CLI:

```shell
python cli.py [--load_data] [--train] [--save_model] [--load_model] [--predict] [--display_graphs] [--load_test_data]
```
Choose the desired options by including the respective command-line arguments:

--load_data: Load the data.

--train: Train the model.

--save_model: Save the model.

--load_model: Load the model.

--predict: Perform predictions.

--display_graphs: Display graphs.

--load_test_data: Load test data (added option).

## Usage Example
```
python cli.py --load_model --load_test_data --predict --display_graphs
```
Follow the prompts and instructions provided by the CLI to interact with the system and perform the desired actions.

Note: You can choose one or more options at a time based on your 
requirements.


## Output

### Model.py
```
The output of running `model.py` is as follows:

          precision    recall  f1-score   support

       0       1.00      1.00      1.00       323
       1       1.00      0.50      0.67        10
       2       0.00      0.00      0.00         3
       3       1.00      0.92      0.96        12
       4       1.00      0.80      0.89         5
       5       1.00      1.00      1.00      1179
       6       1.00      0.43      0.60         7
       7       0.00      0.00      0.00         1
       8       0.00      0.00      0.00         2
       9       1.00      1.00      1.00     13756
      10       0.99      0.99      0.99       517
      11       1.00      1.00      1.00     22018
      12       1.00      1.00      1.00         1
      14       1.00      1.00      1.00        52
      15       1.00      0.99      1.00       946
      16       0.00      0.00      0.00         3
      17       1.00      0.99      0.99      1253
      18       1.00      1.00      1.00       893
      19       0.00      0.00      0.00         1
      20       1.00      1.00      1.00       299
      21       0.99      0.99      0.99       288
      22       0.67      0.67      0.67         3

accuracy                           1.00     41572

macro avg 0.76 0.69 0.72 41572
weighted avg 1.00 1.00 1.00 41572

ROC AUC Score: 0.0019892029280454645

Top 10 important features:
feature importance
4 src_bytes 0.145772
28 same_srv_rate 0.092955
3 flag 0.074988
37 dst_host_serror_rate 0.056490
25 srv_serror_rate 0.056226
33 dst_host_same_srv_rate 0.049757
24 serror_rate 0.049099
29 diff_srv_rate 0.044063
38 dst_host_srv_serror_rate 0.042351
22 count 0.039756
```

### Cli.py

The output of running `cli.py` with the provided command-line arguments is as follows:

```Model loaded.
Loading the model...
Loading the test dataset...
Shape of X_test: (22544, 41)
Test dataset loaded.
Performing predictions...
Predictions completed.
[9 9 11 ... 11 11 9]

              precision    recall  f1-score   support

           0       0.00      0.00      0.00       737
           1       0.00      0.00      0.00       359
           2       0.00      0.00      0.00        20
           3       0.00      0.00      0.00         3
           4       0.00      0.00      0.00      1231
           5       0.00      0.00      0.00       133
           6       0.00      0.00      0.00         1
           7       0.00      0.00      0.00       141
           8       0.00      0.00      0.00         7
           9       0.00      0.00      0.00         2
          10       0.00      0.00      0.00       293
          11       0.02      0.29      0.04       996
          12       0.00      0.00      0.00        18
          13       0.00      0.00      0.00        17
          14       0.00      0.00      0.00      4657
          15       0.00      0.00      0.00        73
          16       0.00      0.00      0.00      9711
          17       0.00      0.00      0.00         2
          18       0.00      0.00      0.00         2
          19       0.00      0.00      0.00        41
          20       0.00      0.00      0.00       157
          21       0.00      0.00      0.00       685
          22       0.00      0.00      0.00        15
          23       0.00      0.00      0.00        13
          24       0.00      0.00      0.00       319
          25       0.00      0.00      0.00       735
          26       0.00      0.00      0.00        14
          27       0.00      0.00      0.00       665
          28       0.00      0.00      0.00       178
          29       0.00      0.00      0.00       331
          30       0.00      0.00      0.00         2
          31       0.00      0.00      0.00        12
          32       0.00      0.00      0.00         2
          33       0.00      0.00      0.00       944
          34       0.00      0.00      0.00         2
          35       0.00      0.00      0.00         9
          36       0.00      0.00      0.00         4
          37       0.00      0.00      0.00        13

    accuracy                           0.01     22544
   macro avg       0.00      0.01      0.00     22544
weighted avg       0.00      0.01      0.00     22544


Displaying graphs...
```
![Graph 1](https://i.imgur.com/koB7Owm.png)


## About the Author
This project is developed by Aditya Varma. I am currently pursuing a Bachelor's degree in Computer Science at the University of Wollongong, majoring in Cybersecurity and AI/Big Data. This project serves as a portfolio project showcasing my skills and interests in the field of intrusion detection and machine learning.

For more information, you can contact me at adityavo@icloud.com.

## License

Copyright (c) 2021-2023 Aditya Varma

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
