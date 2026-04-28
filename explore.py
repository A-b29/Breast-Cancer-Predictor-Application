import pandas as pd
import numpy as np

from logistic_model import LogisticRegressionScratch

df = pd.read_csv("breast cancer wisconsin dataset.csv")

print(df.head())
print(df.info())
print(df.describe)

df["diagnosis"] = df["diagnosis"].map({"M":0 , "B":1})
print(df["diagnosis"].head())
print(df["diagnosis"].describe) 

print(df["diagnosis"].value_counts())

df = df.drop(columns=["id"], errors='ignore')
df = df.drop(columns=["Unnamed: 32"], errors='ignore')


import matplotlib.pyplot as plt

df["diagnosis"].value_counts().plot(kind='pie')
plt.title("Cancer distribution")


df.hist(figsize=(8,6))


print(df["diagnosis"].unique())

corr = df.select_dtypes(include=['number']).corr()
print(corr)

labels = list(corr.columns)

plt.figure(figsize=(12,10))
plt.imshow(corr,cmap='coolwarm')
plt.colorbar()

plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)

plt.title("Correlation Matrix")
plt.tight_layout()
# plt.show()

df["diagnosis"] = 1 - df["diagnosis"]
print(df.corr()["diagnosis"].sort_values(ascending=False))

print("SHAPE:", df.shape)
print("\nCOLUMNS:\n", df.columns)
print("\nFIRST ROW:\n", df.iloc[0])


features = ["concave points_worst", "perimeter_worst", "radius_worst", "perimeter_mean", "area_worst", "radius_mean", "area_mean", "concave points_mean", "concavity_worst", "concavity_mean", "fractal_dimension_worst", "fractal_dimension_mean"]

X = df[features].values
y = df["diagnosis"].values

# Save feature bounds (min, max, mean) for Streamlit app
bounds = np.vstack((np.min(X, axis=0), np.max(X, axis=0), np.mean(X, axis=0)))
np.save("feature_bounds.npy", bounds)

# Step 4: Shuffle
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# Step 5: Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]  


print(np.mean(X_train, axis=0))
print(np.std(X_train, axis=0))


mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

np.save("mean.npy", mean)
np.save("std.npy", std)


model = LogisticRegressionScratch(lr=0.005, epochs=3000)
model.fit(X_train, y_train)
preds = np.array(model.predict(X_test))
accuracy = np.mean(preds == y_test)

np.save("weights.npy", model.weights)
np.save("bias.npy", model.bias)

print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print("Shape of preds:", preds.shape)

tp = np.sum((preds == 1) & (y_test == 1))
fn = np.sum((preds == 0) & (y_test == 1))
fp = np.sum((preds == 1) & (y_test == 0))

recall = tp / (tp + fn)
precision = tp / (tp + fp)



print("Accuracy:", accuracy)

print("Recall:", recall)

print("Precision : ", precision)


import matplotlib.pyplot as plt

plt.plot(model.losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.show()