import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from utils import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

CREDIT_CARD_FILE = "../input/creditcard.csv"

x_items = ["V" + str(i) for i in range(1, 29)] + ["Amount", "Time"]

# Split the data set
data_set = pd.read_csv(CREDIT_CARD_FILE)
Fraud = data_set[data_set.Class == 1]
Normal = data_set[data_set.Class == 0]
Fraud_X = Fraud.filter(items=x_items)
Fraud_Y = Fraud.Class
Normal_X = Normal.filter(items=x_items)
Normal_Y = Normal.Class

# Create training and test sets
train_fraud_X, test_fraud_X, train_fraud_Y, test_fraud_Y = train_test_split(
    Fraud_X, Fraud_Y, test_size=0.4, random_state=0)
train_normal_X, test_normal_X, train_normal_Y, test_normal_Y = \
    train_test_split(Normal_X, Normal_Y, test_size=0.4, random_state=0)

# Repeat data set for fraud classes
temp_X = train_fraud_X.copy()
temp_Y = train_fraud_Y.copy()
for _ in range(200):
   temp_X = pd.concat([temp_X, train_fraud_X])
   temp_Y = pd.concat([temp_Y, train_fraud_Y])

train_X = pd.concat([temp_X, train_normal_X])
train_Y = pd.concat([temp_Y, train_normal_Y])
test_X = pd.concat([test_fraud_X, test_normal_X])
test_Y = pd.concat([test_fraud_Y, test_normal_Y])

# Train
tp = TPOTClassifier(generations=5, population_size=10, verbosity=2)
tp.fit(train_X, train_Y)

# Performances
test_predicted_Y = tp.predict(test_X)
test_score = tp.score(test_X, test_Y)
test_fraud_predicted_Y = tp.predict(test_fraud_X)
test_fraud_score = tp.score(test_fraud_X, test_fraud_Y)

print("Test Set Accuracy: {}".format(test_score))
print("Fraud Classification Accuracy: {}".format(test_fraud_score))

test_confusion = confusion_matrix(test_Y, test_predicted_Y)
test_classes = ["Regular", "Fraud"]
test_title = "Confusion Matrix Test Set. Accuracy {}".format(test_score * 100)
plot_confusion_matrix(test_confusion, test_classes, test_title)
