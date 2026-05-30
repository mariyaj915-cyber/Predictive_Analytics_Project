import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# sample data (later replace your dataset)
data = {
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [2,4,5,4,5,7,8,9,10,12]
}

df = pd.DataFrame(data)

X = df[["x"]]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Graph
plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted")
plt.savefig("images/graph1.png")
plt.close()

# Save model
joblib.dump(model, "models/trained_model.pkl")

print("TRAINING COMPLETED SUCCESSFULLY")
print("y_test:", y_test)
print("y_pred:", y_pred)
plt.figure()
plt.plot(y_test.values, marker='o', label="Actual")
plt.plot(y_pred, marker='x', label="Predicted")
plt.legend()
plt.title("Trend Comparison")
plt.savefig("images/final_trend.png")
plt.close()