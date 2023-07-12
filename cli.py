import argparse

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



class IntrusionDetectionCLI:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CLI for the Intrusion Detection System')
        self.parser.add_argument('--load_data', action='store_true', help='Load the data')
        self.parser.add_argument('--train', action='store_true', help='Train the model')
        self.parser.add_argument('--save_model', action='store_true', help='Save the model')
        self.parser.add_argument('--load_model', action='store_true', help='Load the model')
        self.parser.add_argument('--predict', action='store_true', help='Perform predictions')
        self.parser.add_argument('--display_graphs', action='store_true', help='Display graphs')
        self.parser.add_argument('--load_test_data', action='store_true', help='Load test data')  # Added this line
        self.args = self.parser.parse_args()

        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.encoders = None
        self.imputer = None
        self.scaler = None

    def load_data(self):
        # Load the dataset
        print("Loading the dataset...")
        self.dataset = pd.read_csv("KDDTrain+.txt")
        print("Dataset loaded.")

        # Prepare the dataset
        self.dataset = self.dataset.dropna()

        # Encoding categorical features
        for col in self.dataset.columns:
            if self.dataset[col].dtype == 'object':
                encoder = LabelEncoder()
                self.dataset[col] = encoder.fit_transform(self.dataset[col])

        # Split the dataset into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1], test_size=0.33, random_state=7)

        # Apply imputation
        imputer = SimpleImputer(strategy='mean')
        self.X_train = imputer.fit_transform(self.X_train)
        self.X_test = imputer.transform(self.X_test)

        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def load_test_data(self):
        print("Loading the test dataset...")

        # Define column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
            'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'level'
        ]

        self.X_test = pd.read_csv("KDDTest+.txt", header=None, names=columns)
        self.X_test = self.X_test.drop(['level'], axis=1)  # Drop the 'level' column
        self.y_test = self.X_test['attack_type']  # Save the target variable
        self.X_test = self.X_test.drop(['attack_type'], axis=1)  # Drop the target variable from the features

        print(f"Shape of X_test: {self.X_test.shape}")
        print("Test dataset loaded.")

        # Prepare the test dataset
        self.X_test = self.X_test.dropna()

        # Encoding categorical features in X_test
        for col in ['protocol_type', 'service', 'flag']:
            encoder = LabelEncoder()
            self.X_test[col] = encoder.fit_transform(self.X_test[col])

        # Encode the target variable
        encoder = LabelEncoder()
        self.y_test = encoder.fit_transform(self.y_test)

        # Apply imputation
        self.X_test = self.imputer.transform(self.X_test)

        # Scale the features
        self.X_test = self.scaler.transform(self.X_test)

    def train(self):
        # Train the model
        print("Training the model...")
        self.model = RandomForestClassifier(n_estimators=10)
        self.model.fit(self.X_train, self.y_train)
        print("Model trained.")

    def save_model(self):
        # Save the model
        print("Saving the model...")
        joblib.dump(self.model, 'model.pkl')
        print("Model saved.")

    def load_model(self):
        # Load the model
        print("Loading the model...")
        self.model = joblib.load('model.pkl')
        self.encoders = joblib.load('encoders.pkl')
        self.imputer = joblib.load('imputer.pkl')
        self.scaler = joblib.load('scaler.pkl')
    print("Model loaded.")

    def predict(self):
        print("Performing predictions...")
        y_pred = self.model.predict(self.X_test)
        print("Predictions completed.")
        print(y_pred)

        # Display metrics
        print(classification_report(self.y_test, y_pred))

        # # Convert the predictions to binary form
        # y_score = label_binarize(y_pred, classes=self.encoders['attack_type'].classes_)
        #
        # # Calculate the ROC AUC score
        # avg_roc_auc = roc_auc_score(label_binarize(self.y_test, classes=self.encoders['attack_type'].classes_), y_score,
        #                             average='macro')
        # print("ROC AUC Score: ", avg_roc_auc)

    def display_graphs(self):
        # Display graphs
        print("Displaying graphs...")

        # Graph showing the distribution of attack types
        plt.figure(figsize=(10, 5))
        sns.countplot(data=pd.DataFrame({'attack_type': self.y_test}), x='attack_type')
        plt.title('Distribution of Attack Types')
        plt.show()


    def run(self):
        if self.args.load_data:
            self.load_data()

        if self.args.train:
            self.train()

        if self.args.save_model:
            self.save_model()

        if self.args.load_model or not self.model:
            self.load_model()

        if self.args.load_test_data or not self.X_test:
            self.load_test_data()

        if self.args.predict:
            if self.model and self.X_test is not None:
                self.predict()
            else:
                print("No model or test data available for predictions. Load or train a model and load test data first.")

        if self.args.display_graphs:
            self.display_graphs()


if __name__ == "__main__":
    cli = IntrusionDetectionCLI()
    cli.run()
