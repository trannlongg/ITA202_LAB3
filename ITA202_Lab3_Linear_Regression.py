
# ==========================================
# ITA202 - LAB 3: LINEAR REGRESSION
# HOI QUY TUYEN TINH
# ==========================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# BAI 1: XAY DUNG MO HINH HOI QUY TUYEN TINH
# ==========================================

print("\n========== BAI 1 ==========")

# Tao du lieu gia lap: y = 2X + 1 + nhieu
np.random.seed(42)

X = 2 * np.random.rand(100, 1)
noise = np.random.randn(100, 1)
y = 2 * X + 1 + noise

# Tao mo hinh Linear Regression
model = LinearRegression()

# Huan luyen mo hinh
model.fit(X, y)

# In cac he so
print("Coefficient:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])

# Du doan
y_pred = model.predict(X)

# Ve bieu do
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="blue", label="Du lieu thuc te")
plt.plot(X, y_pred, color="red", label="Duong hoi quy")

plt.title("Bai 1 - Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()


# ==========================================
# BAI 2: DANH GIA MO HINH VOI DIABETES
# ==========================================

print("\n========== BAI 2 ==========")

# Load dataset Diabetes
diabetes = load_diabetes()

# Chi lay 1 dac trung BMI
bmi_index = list(diabetes.feature_names).index("bmi")

X_diabetes = diabetes.data[:, bmi_index].reshape(-1, 1)
y_diabetes = diabetes.target

# Chia du lieu Train va Test
X_train, X_test, y_train, y_test = train_test_split(
    X_diabetes,
    y_diabetes,
    test_size=0.2,
    random_state=42
)

# Tao mo hinh
diabetes_model = LinearRegression()

# Huan luyen
diabetes_model.fit(X_train, y_train)

# Du doan tren tap Test
y_test_pred = diabetes_model.predict(X_test)

# Tinh cac chi so
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R-squared:", r2)

# Giai thich:
# MAE: Sai so tuyet doi trung binh.
# MAE cang nho thi du doan cang gan thuc te.
#
# MSE: Sai so binh phuong trung binh.
# MSE cang nho thi mo hinh cang tot.
#
# R-squared: Do phu hop cua mo hinh.
# R2 cang gan 1 thi mo hinh cang phu hop.
# R2 am nghia la mo hinh du doan kem.


# ==========================================
# BAI 3: TRUC QUAN HOA KET QUA
# ==========================================

print("\n========== BAI 3 ==========")

# Sap xep du lieu de ve duong hoi quy
sort_index = np.argsort(X_diabetes[:, 0])

X_sorted = X_diabetes[sort_index]

y_line = diabetes_model.predict(X_sorted)

# Ve bieu do Scatter Plot
plt.figure(figsize=(9, 5))

plt.scatter(
    X_diabetes,
    y_diabetes,
    color="blue",
    alpha=0.5,
    label="Du lieu thuc te"
)

# Ve duong hoi quy
plt.plot(
    X_sorted,
    y_line,
    color="red",
    linewidth=2,
    label="Duong hoi quy"
)

plt.title("Bai 3 - Diabetes Linear Regression")
plt.xlabel("BMI")
plt.ylabel("Diabetes Target")
plt.legend()
plt.grid()
plt.show()


# ==========================================
# BAI 4: CALIFORNIA HOUSING
# LOAD -> SCALING -> TRAIN -> EVALUATE
# ==========================================

print("\n========== BAI 4 ==========")

# Load du lieu California Housing
housing = fetch_california_housing()

X_housing = housing.data
y_housing = housing.target

# Chia du lieu Train va Test
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_housing,
    y_housing,
    test_size=0.2,
    random_state=42
)

# Scaling va tao mo hinh Linear Regression
housing_model = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

# Huan luyen mo hinh
housing_model.fit(X_train_h, y_train_h)

# Du doan
y_pred_h = housing_model.predict(X_test_h)

# Danh gia bang MSE
mse_housing = mean_squared_error(y_test_h, y_pred_h)

print("MSE California Housing:", mse_housing)

# Truc quan hoa 100 ket qua dau tien
plt.figure(figsize=(10, 5))

plt.plot(
    y_test_h[:100],
    marker="o",
    linestyle="",
    label="Gia tri thuc te"
)

plt.plot(
    y_pred_h[:100],
    marker="x",
    linestyle="",
    label="Gia tri du doan"
)

plt.title("Bai 4 - California Housing")
plt.xlabel("Thu tu mau")
plt.ylabel("Gia nha (100.000 USD)")
plt.legend()
plt.grid()
plt.show()


# ==========================================
# KET THUC LAB 3
# ==========================================

print("\n========== HOAN THANH LAB 3 ==========")
