# predicción del precio de un coche de segunda mano utilizando la regresión lineal 
# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Charger les données depuis le fichier CSV
data = pd.read_csv("car data.csv")
X = data["Year"].values.reshape(-1, 1)  # Année de fabrication (indépendante)
y = data["Selling_Price"].values # Prix de vente (dépendante)
print(data.head())



#Ajuster le modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Coefficients de régression
b0 = model.intercept_  # Intercept (b0)
b1 = model.coef_[0]    # Pente (b1)

print(f"Intercept (b0) : {b0}")
print(f"Slope (b1) : {b1}")


#Prédiction pour une voiture de l'année 2025
year_to_predict = np.array([[2025]])
predicted_price = model.predict(year_to_predict)
print(f"Prix prédit pour 2025 : {predicted_price[0]:.2f} ")

#Histogramme du Prix de vente 
plt.figure(figsize=(8, 5))
plt.hist(data['Selling_Price'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution du Prix de Vente")
plt.xlabel("Prix de Vente")
plt.ylabel("Nombre de Voitures")
plt.grid(True)
plt.show()

#Graphique en Ligne du Prix de Vente par Année
average_price_by_year = data.groupby('Year')['Selling_Price'].mean()

plt.figure(figsize=(10, 6))
plt.plot(average_price_by_year.index, average_price_by_year.values, marker='o', linestyle='-', color='green')
plt.title("Prix de Vente Moyen par Année")
plt.xlabel("Année de Fabrication")
plt.ylabel("Prix de Vente Moyen")
plt.grid(True)
plt.show()

#Diagramme en Boîte des Prix par Type de Carburant
plt.figure(figsize=(10, 6))
data.boxplot(column='Selling_Price', by='Fuel_Type', grid=True)
plt.title("Prix de Vente par Type de Carburant")
plt.xlabel("Type de Carburant")
plt.ylabel("Prix de Vente")
plt.suptitle("")  # Supprimer le titre par défaut de matplotlib
plt.show()


#Nuage de Points des Kilomètres Parcourus vs Prix de Vente
plt.figure(figsize=(10, 6))
plt.scatter(data['Kms_Driven'], data['Selling_Price'], color='purple', alpha=0.5)
plt.title("Kilomètres Parcourus vs Prix de Vente")
plt.xlabel("Kilomètres Parcourus")
plt.ylabel("Prix de Vente")
plt.grid(True)
plt.show()


# Diagramme à Barres du Nombre de Voitures par Année
car_count_by_year = data['Year'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
car_count_by_year.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Nombre de Voitures par Année de Fabrication")
plt.xlabel("Année de Fabrication")
plt.ylabel("Nombre de Voitures")
plt.grid(axis='y')
plt.show()


#Courbe de Régression pour le Prix en Fonction des Kilomètres Parcourus
X_kms = data['Kms_Driven'].values.reshape(-1, 1)
y_price = data['Selling_Price'].values

model_kms = LinearRegression()
model_kms.fit(X_kms, y_price)

plt.figure(figsize=(10, 6))
plt.scatter(X_kms, y_price, color='blue', alpha=0.5, label='Données réelles')
plt.plot(X_kms, model_kms.predict(X_kms), color='red', label='Droite de régression')
plt.title("Régression Linéaire - Prix de Vente vs Kilomètres Parcourus")
plt.xlabel("Kilomètres Parcourus")
plt.ylabel("Prix de Vente")
plt.legend()
plt.grid(True)
plt.show()


#Boxplot (Diagramme en Boîte) : Prix de Vente par Année
plt.figure(figsize=(12, 6))
sns.boxplot(x='Year', y='Selling_Price', data=data, hue='Year', palette='coolwarm', legend=False)
plt.title("Prix de Vente par Année de Fabrication (Boxplot)")
plt.xlabel("Année de Fabrication")
plt.ylabel("Prix de Vente")
plt.xticks(rotation=45)
plt.show()


# Violin plot avec subplots
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

sns.violinplot(x='Fuel_Type', y='Selling_Price', data=data, ax=ax[0], hue='Fuel_Type', palette='muted', legend=False)
ax[0].set_title("Prix de Vente par Type de Carburant")

sns.violinplot(x='Transmission', y='Selling_Price', data=data, ax=ax[1], hue='Transmission', palette='muted', legend=False)
ax[1].set_title("Prix de Vente par Type de Transmission")

plt.tight_layout()
plt.show()


#Pairplot avec Diagonal KDE
sns.pairplot(data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven']], diag_kind='kde', plot_kws={'color': 'purple'})
plt.show()


#Hexbin Plot : Nombre de Kilomètres vs Prix de Vente
plt.figure(figsize=(10, 6))
plt.hexbin(data['Kms_Driven'], data['Selling_Price'], gridsize=50, cmap='Blues')
plt.title("Hexbin Plot : Kilomètres Parcourus vs Prix de Vente")
plt.xlabel("Kilomètres Parcourus")
plt.ylabel("Prix de Vente")
plt.colorbar(label='Densité')
plt.show()