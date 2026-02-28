# ==========================================
# HOUSE PRICE PREDICTION - FULL PIPELINE
# ==========================================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error


# ==========================================
# 2️⃣ LOAD TRAIN DATA
# ==========================================

train_df = pd.read_csv("train.csv")

print("Train Shape:", train_df.shape)

# Drop ID
train_df.drop("Id", axis=1, inplace=True)

# Log transform target
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# Separate target
y = train_df["SalePrice"]
X = train_df.drop("SalePrice", axis=1)


# ==========================================
# 3️⃣ HANDLE MISSING VALUES
# ==========================================

# Numeric
for col in X.select_dtypes(include=np.number):
    X[col] = X[col].fillna(X[col].median())

# Categorical
for col in X.select_dtypes(include="object"):
    X[col] = X[col].fillna("None")


# ==========================================
# 4️⃣ ONE HOT ENCODING
# ==========================================

X = pd.get_dummies(X, drop_first=True)

print("Shape after encoding:", X.shape)


# ==========================================
# 5️⃣ TRAIN TEST SPLIT (FOR LOCAL CHECK)
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Using Ridge (better than simple Linear Regression)
model = Ridge(alpha=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Performance (Validation Set)")
print("R2 Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# ==========================================
# 6️⃣ PLOT ACTUAL VS PREDICTED
# ==========================================

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Price (log)")
plt.ylabel("Predicted Price (log)")
plt.title("Actual vs Predicted")
plt.show()


# ==========================================
# 7️⃣ LOAD TEST DATA (KAGGLE TEST FILE)
# ==========================================

test_df = pd.read_csv("test.csv")

print("Test Shape:", test_df.shape)

# Save Id for submission
test_ids = test_df["Id"]

# Drop Id
test_df.drop("Id", axis=1, inplace=True)


# ==========================================
# 8️⃣ HANDLE TEST MISSING VALUES
# ==========================================

for col in test_df.select_dtypes(include=np.number):
    test_df[col] = test_df[col].fillna(test_df[col].median())

for col in test_df.select_dtypes(include="object"):
    test_df[col] = test_df[col].fillna("None")


# ==========================================
# 9️⃣ ENCODE TEST DATA
# ==========================================

test_df = pd.get_dummies(test_df, drop_first=True)

# Align test columns with training columns
test_df = test_df.reindex(columns=X.columns, fill_value=0)


# ==========================================
# 🔟 TRAIN ON FULL TRAIN DATA
# ==========================================

model.fit(X, y)

# Predict test data
test_predictions = model.predict(test_df)

# Convert log back to normal price
test_predictions = np.expm1(test_predictions)


# ==========================================
# 1️⃣1️⃣ CREATE SUBMISSION FILE
# ==========================================

submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("\nSubmission file created successfully!")
print("File name: submission.csv")
