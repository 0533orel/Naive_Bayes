from csv_data_loader import CsvDataLoader
from model_tester import ModelTester

class ConsoleUI:
    """
    Console-based user interface for interacting with the Naive Bayes classifier.

    Provides options for loading data, training the model, testing accuracy,
    and classifying a single record.
    """

    def __init__(self):
        """
        Initializes the ConsoleUI without a loaded model.
        """
        self.model_tester = None

    def run(self):
        """
        Starts the user interaction loop.

        Displays a menu with options:
            1. Load dataset and train the model
            2. Test model accuracy on test data
            3. Classify a single record
            4. Exit

        Responds to user input accordingly.
        """
        while True:
            print("\n----- Naive Bayes Classifier -----")
            print("1. Load dataset and train model")
            print("2. Test model accuracy on test data")
            print("3. Classify a single record")
            print("4. Exit")
            choice = input("\nChoose an option (1-4): ")

            if choice == '1':
                path = input("\nEnter path to CSV file: ")
                target_col = input("\nEnter the name of the target column (leave empty to use the last column): ")
                try:
                    loader = CsvDataLoader(path, target_col)
                    self.model_tester = ModelTester(loader.df)
                    self.model_tester.train_model()
                    print("\nModel trained successfully.")
                except Exception as e:
                    print(f"\nError: {e}")

            elif choice == '2':
                if self.model_tester:
                    successful_answer = self.model_tester.test_model()
                    print(successful_answer)
                else:
                    print("\nYou must train a model first.")

            elif choice == '3':
                if self.model_tester:
                    try:
                        print("\nEnter values for each feature:")
                        sample_dict = {}
                        features = list(self.model_tester.model.x.values())[0].keys()
                        for feature in features:
                            possible_values = list(list(self.model_tester.model.x.values())[0][feature].keys())
                            print(f"\nChoose a value for '{feature}':")
                            for i, val in enumerate(possible_values):
                                print(f"{i + 1}. {val}")
                            while True:
                                try:
                                    choice_index = int(input("\nEnter your choice (number): ")) - 1
                                    if 0 <= choice_index < len(possible_values):
                                        sample_dict[feature] = possible_values[choice_index]
                                        break
                                    else:
                                        print("\nInvalid choice. Please choose a valid number.")
                                except ValueError:
                                    print("\nInvalid input. Please enter a number.")
                        result = self.model_tester.model.predict(sample_dict)
                        print(f"\nPredicted label: {result}")
                    except Exception as e:
                        print(f"\nError during classification: {e}")
                else:
                    print("\nYou must train a model first.")

            elif choice == '4':
                print("\nGoodbye")
                break

            else:
                print("\nInvalid option. Please choose 1-4.")
