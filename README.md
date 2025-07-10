# Naive Bayes Classifier (Console Application)

This project implements a Naive Bayes Classifier with Laplace smoothing using object-oriented programming (OOP) in Python. It is designed for educational purposes and command-line interaction.

## Project Structure

.
├── main.py                  # Entry point for running the application
├── console_UI.py            # Console-based user interface
├── model_tester.py          # Handles model training and testing
├── naive_bayes_classifier.py # The Naive Bayes algorithm implementation
├── csv_data_loader.py       # Responsible for loading and cleaning CSV files
├── requirements.txt         # Python dependencies (optional)
└── README.md                # Project documentation

## Features

- Load and clean datasets from CSV files
- Automatically splits data (70% training / 30% testing)
- Applies Laplace correction to avoid zero probabilities
- Trains a custom Naive Bayes classifier
- Tests model accuracy on unseen data
- Classifies a single record with user input via the terminal

## How to Run

1. Install requirements (if not already installed):

    pip install pandas

2. Run the program:

    python main.py

3. Follow the menu options to load data, train the model, test accuracy, and classify records.

## Example Datasets

You can use any categorical dataset, such as:

- UCI Mushroom Dataset: https://archive.ics.uci.edu/ml/datasets/mushroom
- Phishing Website Dataset (Kaggle): https://www.kaggle.com/eswarchandt/phishing-website-detector

Just download and provide the CSV file path when prompted.

## Requirements

- Python 3.7+
- pandas

To generate a `requirements.txt` file:

    pip freeze > requirements.txt

Then install with:

    pip install -r requirements.txt

## Educational Objectives

- Practice object-oriented design and class structuring
- Gain experience with probability-based classification
- Apply Laplace smoothing effectively
- Develop modular and maintainable code

## Authors

This project was developed as part of an academic assignment.
