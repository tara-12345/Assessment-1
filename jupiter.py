import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Moons:
    def __init__(self):
        self.data = self.load_data() #attributing data set to moons data 
        self.names = self.load_names()
        
    def load_data (self): #opening and reading moons database 
        
        connectable = f"sqlite:///jupiter.db"
        query = "SELECT * FROM moons"
        
        jupiter_data = pd.read_sql(query, connectable)
        jupiter_data.set_index("moon", inplace=True)
        self.jupiter_data = jupiter_data
        
        return jupiter_data

    def load_names(self): #determing names for moons 
        names_list = []
        for names in self.load_data().index:
            names_list.append(names)
            
        return names_list
    
    def plot(self, x, y):
        
        
        plt.scatter(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
    
    def correlation_heatmap(self):
        
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        corr_matrix = self.data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='viridis')
        plt.title('Correlation Heatmap')
        plt.show()
    
    def correlation(self):
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        corr_matrix = self.data.corr()
        
        var_1 = input("Variable 1")
        var_2 = input("Variable 2")
        
        if var_1 and var_2 in self.jupiter_data.iloc[0]:
            variable_correlation = corr_matrix[var_1][var_2]
        
        return variable_correlation
       
        
    def column_average(self):

        column_averages = {'period average' : [self.data["period_days"].mean(axis=0)],
                           'distance average' : [self.data["distance_km"].mean(axis=0)],
                           'radius average' : [self.data["radius_km"].mean(axis=0)],
                           'magnitude average' : [self.data["mag"].mean(axis=0)],
                           'eccentricity average' : [self.data["ecc"].mean(axis=0)],
                           'inclination degree average' : [self.data["inclination_deg"].mean(axis=0)]}
        
        averages_transposed = pd.DataFrame(column_averages).transpose()
        column_average_df = averages_transposed.rename(columns={0: "Average value"})
        
        display (column_average_df)
       
    
    def moon_info(self):

            x = input("name of moon")
            
            if x in self.jupiter_data.index:
                
                moon_info = pd.DataFrame(self.jupiter_data.loc[x])
                return moon_info
    
            

        
        
    def train(self):
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn import linear_model
        
        self.data["radius_variable"] = (self.data["distance_km"]*1000)**3
        self.data["period_variable"] = (self.data["period_days"]*86400)** 2
        
        X = self.data[["radius_variable"]]
        Y = self.data["period_variable"]
        
        model = linear_model.LinearRegression()
        
        x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size =0.3 , random_state=42)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        
        fig, ax = plt.subplots()
        ax.scatter(X,Y)
        ax.plot(x_test, pred)
        
        
        plt.show()
        print(f"The R2 score is: {r2_score(y_test,pred)}")
        
        jupiter_mass = (4 * np.pi**2) / (model.coef_[0]*6.67e-11)
        return jupiter_mass
        
        


        
       