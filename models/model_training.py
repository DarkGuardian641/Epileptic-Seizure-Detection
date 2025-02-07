import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load processed data
processed_file_path = "data/processed_data/Epileptic_Seizure_Processed.csv"
data = pd.read_csv(processed_file_path)

# Ensure there are no missing values
data = data.dropna()

# Separate features and target
X = data.drop(columns=['y'])
y = data['y']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize multiple models
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
    ("Support Vector Machine", SVC(kernel='rbf', probability=True, random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("Naive Bayes", GaussianNB()),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
    ("AdaBoost", AdaBoostClassifier(random_state=42)),
    ("Extra Trees", ExtraTreesClassifier(random_state=42))
]

# Train and evaluate each model
results = []

for name, model in models:
    print(f"Training {name}...")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((name, accuracy))
        print(f"Accuracy for {name}: {accuracy}")
        print(classification_report(y_test, y_pred))
    except Exception as e:
        print(f"Error training {name}: {e}")

# Print summary of all models
print("\nSummary of Model Accuracies:")
for name, accuracy in results:
    print(f"{name}: {accuracy:.4f}")
