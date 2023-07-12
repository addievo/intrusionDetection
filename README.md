# Intrusion Detection System

This project implements an Intrusion Detection System (IDS) using machine learning techniques. It aims to detect and classify network intrusions or anomalous activities in a computer network.

## Project Description

The Intrusion Detection System consists of two main components: the CLI (Command-Line Interface) and the machine learning model. The CLI provides a user-friendly interface to interact with the system, including options to load data, train the model, save and load the model, perform predictions, and display graphs. The machine learning model is built using the Random Forest Classifier algorithm to classify network traffic as normal or anomalous.

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

##For Example
```
python cli.py --load_model --load_test_data --predict --display_graphs
```
Follow the prompts and instructions provided by the CLI to interact with the system and perform the desired actions.

Note: You can choose one or more options at a time based on your 
requirements.

License
This project is licensed under the MIT License.

About the Author
This project is developed by Aditya Varma. I am currently pursuing a Bachelor's degree in Computer Science at the University of Wollongong, majoring in Cybersecurity and AI/Big Data. This project serves as a portfolio project showcasing my skills and interests in the field of intrusion detection and machine learning.

For more information, you can contact me at adityavo@icloud.com.
