# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Étape 1 : Charger les données depuis le fichier CSV
data = pd.read_csv("car data.csv")
X = data["Year"].values.reshape(-1, 1)  # Année de fabrication (indépendante)
y = data["Selling_Price"].values        # Prix de vente (dépendante)

# Étape 2 : Ajuster le modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Coefficients de régression
b0 = model.intercept_  # Intercept (b0)
b1 = model.coef_[0]    # Pente (b1)

print(f"Intercept (b0) : {b0}")
print(f"Slope (b1) : {b1}")

# Étape 3 : Prédiction pour une voiture de l'année 2025
year_to_predict = np.array([[2025]])
predicted_price = model.predict(year_to_predict)
print(f"Prix prédit pour 2025 : {predicted_price[0]:.2f} ")

# Étape 4 : Visualisation des résultats
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X, model.predict(X), color='red', label='Droite de régression')
plt.title("Régression Linéaire - Prix de vente vs Année de fabrication")
plt.xlabel("Année de fabrication")
plt.ylabel("Prix de vente")
plt.legend()
plt.grid(True)
plt.show()
