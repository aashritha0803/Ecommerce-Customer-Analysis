# E-COMMERCE CUSTOMER SHOPPING ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load dataset
data = pd.read_csv("C:\\Users\\91984\\Downloads\\ecommerce_customer_data.csv")

print("Top five rows")
print(data.head())

print("Bottom five rows")
print(data.tail())

print("\nDATASET INFORMATION")
print(data.info())

print("\nSUMMARY STATISTICS")
print(data.describe(include="all"))

# 2. Check missing values and duplicates
print("\nMISSING VALUES")
print(data.isnull().sum())

print("\nDUPLICATE ROWS")
print(data.duplicated().sum())

# 3. Data cleaning
data = data.drop_duplicates().copy()

# Fill missing categorical values with the most frequent value
for column in ["Gender", "Payment_Method", "City"]:
    data[column] = data[column].fillna(data[column].mode()[0])

# Fill missing age with the median
data["Age"] = data["Age"].fillna(data["Age"].median())

# Convert order date to datetime
data["Order_Date"] = pd.to_datetime(data["Order_Date"], errors="coerce")
data = data.dropna(subset=["Order_Date"])

# Check data types after cleaning
print("\nMISSING VALUES AFTER CLEANING")
print(data.isnull().sum())

# 4. Createanalysis columns
data["Revenue"] = data["Quantity"] * data["Price"]
data["Month"] = data["Order_Date"].dt.month_name()
data["Month_Number"] = data["Order_Date"].dt.month
data["Year"] = data["Order_Date"].dt.year

# 5. Key performance indicators (KPIs)
total_revenue = data["Revenue"].sum()
total_orders = data["Order_ID"].nunique()
total_customers = data["Customer_ID"].nunique()
average_order_value = total_revenue / total_orders

print("\nKEY PERFORMANCE INDICATORS")
print(f"Total Revenue: Rs. {total_revenue:,.2f}")
print(f"Total Orders: {total_orders}")
print(f"Unique Customers: {total_customers}")
print(f"Average Order Value: Rs. {average_order_value:,.2f}")

# 6. Revenue by product category
category_revenue = (
    data.groupby("Product_Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)
print("\nREVENUE BY CATEGORY")
print(category_revenue)

plt.figure(figsize=(9, 5))
sns.barplot(x=category_revenue.index, y=category_revenue.values)
plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("01_revenue_by_category.png")
plt.show()

# 7. Top 10 selling products by quantity
top_products = (
    data.groupby("Product")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print("\nTOP 10 PRODUCTS BY QUANTITY")
print(top_products)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("Top 10 Selling Products")
plt.xlabel("Units Sold")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("02_top_products.png")
plt.show()

# 8. Monthly revenue trend
monthly_revenue = (
    data.groupby(["Year", "Month_Number"])["Revenue"]
    .sum()
    .reset_index()
    .sort_values(["Year", "Month_Number"])
)
monthly_revenue["Month_Label"] = (
    monthly_revenue["Year"].astype(str) + "-" +
    monthly_revenue["Month_Number"].astype(str).str.zfill(2)
)

plt.figure(figsize=(11, 5))
sns.lineplot(data=monthly_revenue, x="Month_Label", y="Revenue", marker="o")
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue (Rs.)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("03_monthly_revenue.png")
plt.show()

# 9. Payment method analysis
payment_counts = data["Payment_Method"].value_counts()
print("\nORDERS BY PAYMENT METHOD")
print(payment_counts)

plt.figure(figsize=(7, 7))
plt.pie(payment_counts.values, labels=payment_counts.index,
        autopct="%1.1f%%", startangle=90)
plt.title("Payment Method Distribution")
plt.tight_layout()
plt.savefig("04_payment_methods.png")
plt.show()

# 10. Customer gender analysis
gender_revenue = data.groupby("Gender")["Revenue"].sum()
print("\nREVENUE BY GENDER")
print(gender_revenue)

plt.figure(figsize=(7, 5))
sns.barplot(x=gender_revenue.index, y=gender_revenue.values)
plt.title("Revenue by Gender")
plt.xlabel("Gender")
plt.ylabel("Revenue (Rs.)")
plt.tight_layout()
plt.savefig("05_revenue_by_gender.png")
plt.show()

# 11. Age group analysis
data["Age_Group"] = pd.cut(
    data["Age"],
    bins=[17, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)
age_revenue = data.groupby("Age_Group", observed=False)["Revenue"].sum()
print("\nREVENUE BY AGE GROUP")
print(age_revenue)

plt.figure(figsize=(8, 5))
sns.barplot(x=age_revenue.index, y=age_revenue.values)
plt.title("Revenue by Customer Age Group")
plt.xlabel("Age Group")
plt.ylabel("Revenue (Rs.)")
plt.tight_layout()
plt.savefig("06_revenue_by_age_group.png")
plt.show()

# 12. City-wise revenue
city_revenue = data.groupby("City")["Revenue"].sum().sort_values(ascending=False)
print("\nREVENUE BY CITY")
print(city_revenue)

plt.figure(figsize=(9, 5))
sns.barplot(x=city_revenue.values, y=city_revenue.index)
plt.title("Revenue by City")
plt.xlabel("Revenue (Rs.)")
plt.ylabel("City")
plt.tight_layout()
plt.savefig("07_revenue_by_city.png")
plt.show()

# 13. Correlation heatmap for numeric columns
numeric_data = data[["Age", "Quantity", "Price", "Revenue"]]
plt.figure(figsize=(7, 5))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("08_correlation_heatmap.png")
plt.show()

# 14. Export cleaned dataset
data.to_csv("cleaned_ecommerce_data.csv", index=False)
print("\nAnalysis completed. Cleaned dataset and charts have been saved.")


cleaned_data = pd.read_csv("cleaned_ecommerce_data.csv")

print(cleaned_data.head())
print(cleaned_data.shape)
print(cleaned_data.isnull().sum())
print(cleaned_data.duplicated().sum())