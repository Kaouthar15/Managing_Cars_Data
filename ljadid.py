import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger et nettoyer les données
car_data = pd.read_csv('car data.csv')
car_data['Selling_Price'] = car_data['Selling_Price'].fillna(car_data['Selling_Price'].mean())

# Régression linéaire : Année vs Prix de Vente
X = car_data[['Year']]  # Variable indépendante
y = car_data['Selling_Price']  # Variable dépendante
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Évaluer et visualiser les résultats
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE : {rmse:.2f}")
plt.scatter(X_test, y_test, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', label='Ligne de régression')
plt.xlabel('Année')
plt.ylabel('Prix de Vente')
plt.legend()
plt.title('Régression Linéaire : Prix de Vente vs Année')
plt.show()
