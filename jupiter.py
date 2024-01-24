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
        
        read_df = pd.read_sql(query, connectable)
        jupiter_data = read_df.rename(columns={'period_days': 'Period (days)', 
                                               'distance_km': 'Distance (km)',
                                               'radius_km': 'Radius (km)', 
                                               'mag': 'Mangnitude',
                                               'group' : "Group", 
                                               "mass_kg" : "Mass (kg)",
                                               "ecc" : "Eccentricity", 
                                               "inclination_deg" : "Inlination Degree"})
        jupiter_data.set_index("moon", inplace=True)
        self.jupiter_data = jupiter_data
        return jupiter_data

    def load_names(self): #determing names for moons 
        names_list = []
        for names in self.load_data().index:
            names_list.append(names)
            
        return names_list
    
    def moons_head(self):
        return self.data.head()
    
    def drop_navalues(self, x):
        dropna_df = self.data[self.data[x].notna()]
        return dropna_df
    
    def data_summaries(self):
        return self.data.describe()
    
    def plot(self, x, y):
        sns.relplot(self.data, x=x, y=y, hue="Group")
        plt.title("Plot of Correlation Between Two Variables")
        plt.show()
        
        
    def distributions(self):
        
        number_columns = self.data.drop("Group", axis=1)
        for col in number_columns:
            sns.histplot(self.data, x=col, bins=5)
            plt.title(f"Distribution of {col}")
            plt.show()
            
    
    def correlation_heatmap(self):
        
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        
        corr_matrix = self.data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='viridis')
        plt.title('Correlation Heatmap')
        plt.show()
    
    def correlation_value(self):
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        corr_matrix = self.data.corr()
        
        var_1 = input("Variable 1")
        var_2 = input("Variable 2")
        
        if var_1 and var_2 in self.jupiter_data.iloc[0]:
            variable_correlation = print(f"The correlation value between {var_1} and {var_2} is {corr_matrix[var_1][var_2].round(3)}")
        
        return variable_correlation

        
    def column_average(self):

        column_averages = {'Period Average (days)' : [self.data["Period (days)"].mean(axis=0)],
                           'Distance Average (km)' : [self.data["Distance (km)"].mean(axis=0)],
                           'Radius Average (km)' : [self.data["Radius (km)"].mean(axis=0)],
                           'Magnitude Average' : [self.data["Mangnitude"].mean(axis=0)],
                           'Eccentricity Average' : [self.data["Eccentricity"].mean(axis=0)],
                           'Inclination Degree Average (Degrees)' : [self.data["Inlination Degree"].mean(axis=0)]}
        
        averages_transposed = pd.DataFrame(column_averages).transpose()
        column_average_df = averages_transposed.rename(columns={0: "Average value"})
        column_average_df['Average value'] = column_average_df['Average value']
        
        display (column_average_df)
       
    
    def moon_info(self):

            x = input("Name of moon for information summary")
            
            if x in self.jupiter_data.index:
                
                moon_info = pd.DataFrame(self.jupiter_data.loc[x])
                return moon_info
            else:
                none = print("None")
                return none
    
    def train(self):
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn import linear_model
        
        self.data["radius_variable"] = (self.data["Distance (km)"]*1000)**3
        self.data["period_variable"] = (self.data["Period (days)"]*86400)** 2
        
        X = self.data[["radius_variable"]]
        Y = self.data["period_variable"]
        
        model = linear_model.LinearRegression()
            
        x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size =0.3 , random_state=42)

        
        model.fit(x_train, y_train)
        
        training_pred_y = model.predict(x_train)
        testing_pred_y = model.predict(x_test)
        
        plt.figure()
        plt.scatter(X, Y)
        plt.plot(x_test, testing_pred_y)
        plt.show()
        
        jupiter_mass = (4 * np.pi**2) / (model.coef_[0]*6.67e-11)
        print(f"The R2 score is: {r2_score(y_test,testing_pred_y)}")
        
        training_res = y_train - training_pred_y
        testing_residuals = y_test - testing_pred_y
        
        plt.scatter(x_train, training_res, color='red', label='Testing Residuals')
        plt.title("Plot of Residuals on the Training set")
        plt.show()
       
        return print(f"The predicted masss of jupiter is {jupiter_mass}kg.")
    
    
 
    
    
        

      
        


        
       