import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('data/data_company.trn', sep=';')

if not dataset.isnull().values.any():
    print('There are no missing values in our dataset, so no need to fill in any missing data.')

# Get all features
x = dataset.iloc[:, 2:].values

# Get the dependent variable
y = dataset.iloc[:, 1:2].values

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

# Scale the features
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

# Fitting classifier to the Training set
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(cm)
print('Accuracy: ' + str(score*100) + '%')