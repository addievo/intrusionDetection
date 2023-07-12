import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import label_binarize

# fix random seed for reproducibility
np.random.seed(7)

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

# load KDDCUP99 dataset
dataset = pd.read_csv("KDDTrain+.txt", delimiter=",", header=None, names=columns)
dataset = dataset.sample(frac=1)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(dataset['attack_type'])
encoded_Y = encoder.transform(dataset['attack_type'])

# encode categorical features as integers
for col in ['protocol_type', 'service', 'flag']:
    encoder = LabelEncoder()
    dataset[col] = encoder.fit_transform(dataset[col])
# encode categorical features as integers
encoders = {}  # Create a dictionary to store the encoders
for col in ['protocol_type', 'service', 'flag']:
    encoder = LabelEncoder()
    dataset[col] = encoder.fit_transform(dataset[col])
    encoders[col] = encoder  # Store the encoder in the dictionary

# Save encoders
joblib.dump(encoders, 'encoders.pkl')

# split into 67% for train and 33% for test
train, test, trainlabel, testlabel = train_test_split(dataset.drop(['attack_type', 'level'], axis=1), encoded_Y,
                                                      test_size=0.33, random_state=7)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
train_imputed = imputer.fit_transform(train)
train = pd.DataFrame(train_imputed, columns=train.columns)
test = imputer.transform(test)

# create model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Fit the model
model.fit(train, trainlabel)



# predictions
predictions = model.predict(test)

# evaluate predictions
print(classification_report(testlabel, predictions))

# ROC AUC Score
testlabel_bin = label_binarize(testlabel, classes=np.unique(testlabel))
avg_roc_auc = roc_auc_score(testlabel_bin, predictions.reshape(-1, 1), average='macro')
print("ROC AUC Score: ", avg_roc_auc)

# feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({'feature': train.columns, 'importance': importances})
feature_importance.sort_values(by='importance', ascending=False, inplace=True)
print(feature_importance.head(10))

# Save model
joblib.dump(model, 'model.pkl')

# Save imputer and encoder
joblib.dump(imputer, 'imputer.pkl')

#Save scaler
scaler = StandardScaler()
train = scaler.fit_transform(train)

joblib.dump(scaler, 'scaler.pkl')

#saving encoder
