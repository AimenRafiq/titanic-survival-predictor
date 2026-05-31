import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset directly from the web
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Total passengers: {len(df)}")
print(f"Survived: {len(df[df['Survived'] == 1])}")
print(f"Did not survive: {len(df[df['Survived'] == 0])}")
print()

# Step 2: Pick the columns we want the model to learn from
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
df = df[features + ["Survived"]].dropna()

# Step 3: Convert "male/female" to numbers (0 and 1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Step 4: Split into training and testing
X = df[features]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.1f}%")
print()
print(classification_report(y_test, predictions))

# Step 7: See which features mattered most
print("--- Feature Importance ---")
for feature, importance in zip(features, model.feature_importances_):
    print(f"{feature}: {importance * 100:.1f}%")