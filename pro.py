import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the car data from the CSV file
car_data = pd.read_csv('car data.csv')

# Let's take a look at the basic details of the dataset
print("Here's the dataset info:")
print(car_data.info())
print("\nLet's preview the first few rows:")
print(car_data.head())

# Explore the summary statistics for a quick overview of the data
print("\nSummary statistics of the dataset:")
print(car_data.describe())

# Clean data, Find out if there are any duplicate rows in the data
duplicates = car_data.duplicated().sum()
print(f"\nWe found {duplicates} duplicate rows.")

# Check for any missing values in the dataset
missing_values = car_data.isnull().sum()
print("\nHere's the count of missing values in each column:")
print(missing_values)

# Handling missing values by filling them with the average value for 'Selling_Price'
car_data['Selling_Price'] = car_data['Selling_Price'].fillna(np.mean(car_data['Selling_Price']))
print("\nMissing values in 'Selling_Price' have been filled with the mean.")

#  Let’s visualize some aspects of the data

# Plot the distribution of fuel types in the dataset
plt.figure(figsize=(8, 5))
sns.countplot(x='Fuel_Type', data=car_data)
plt.title('Distribution of Fuel Types')
plt.show()

# Explore the relationship between the year of the car and its selling price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Selling_Price', data=car_data)
plt.title('Selling Price vs. Year of the Car')
plt.show()

# Normalize the 'Selling_Price' column to scale values between 0 and 1
selling_price = car_data['Selling_Price'].values  # Convert the column to a NumPy array for easier manipulation
normalized_price = (selling_price - np.min(selling_price)) / (np.max(selling_price) - np.min(selling_price))
car_data['Normalized_Selling_Price'] = normalized_price

# Convert categorical data into numeric format using one-hot encoding
car_data_encoded = pd.get_dummies(car_data, drop_first=True)
print("\nHere’s the data after encoding categorical variables:")
print(car_data_encoded.head())

# Save the cleaned and encoded data to a new CSV file
car_data_encoded.to_csv('car_data_cleaned.csv', index=False)
print("\nThe cleaned data has been successfully saved to 'car_data_cleaned.csv'!")
