# src/train.py
import joblib
from sklearn.tree import DecisionTreeClassifier

# Training data: age as feature, 1 for adult, 0 for minor
X = [[10], [15], [17], [18], [20], [25], [30]]
y = [0, 0, 0, 1, 1, 1, 1]

model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "age_model.pkl")
print("Model trained and saved as age_model.pkl")
