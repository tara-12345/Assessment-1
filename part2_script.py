import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt

connectable = f"sqlite:///jupiter.db"
query = "SELECT moon, period_days, distance_km, radius_km, mag, mass_kg, ecc, inclination_deg FROM moons"
jupiter_data = pd.read_sql(query, connectable)
jupiter_data = jupiter_data.set_index("moon")
jupiter_data.info()

radius_cube = (jupiter_data["distance_km"]*1000)**3
time_squared = (jupiter_data["period_days"]*86400)** 2
                         
jupiter_data["radius_variable"] = radius_cube
jupiter_data["period_variable"] = time_squared
print(radius_cube)

model = linear_model.LinearRegression()

X = jupiter_data[["radius_variable"]]
Y = jupiter_data["period_variable"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y , test_size =0.3 , random_state=42)

model.fit(x_train, y_train)
pred = model.predict(x_test)

fig, ax = plt.subplots()
ax.scatter(X,Y)
ax.plot(x_test, pred)