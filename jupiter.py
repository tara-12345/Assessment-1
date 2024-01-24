import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Moons:
    def __init__(self): #initalise data set 
        self.data = self.load_data() #attributing data set to moons data 
        self.names = self.load_names()
        
    def load_data (self): #opening and reading moons database 
        
        connectable = f"sqlite:///jupiter.db"
        query = "SELECT * FROM moons"
        
        read_df = pd.read_sql(query, connectable) #renaming / capitalising column titles
        jupiter_data = read_df.rename(columns={'period_days': 'Period (days)', 
                                               'distance_km': 'Distance (km)',
                                               'radius_km': 'Radius (km)', 
                                               'mag': 'Magnitude',
                                               'group' : "Group", 
                                               "mass_kg" : "Mass (kg)",
                                               "ecc" : "Eccentricity", 
                                               "inclination_deg" : "Inlination Degree"})
        jupiter_data.set_index("moon", inplace=True) #resetting index for easy referral
        self.jupiter_data = jupiter_data
        return jupiter_data

    def load_names(self): #determing names for moons 
        names_list = []
        for names in self.load_data().index:
            names_list.append(names)
            
        return names_list
    
    def moons_head(self): #defining method for reading the first few lines of data
        return self.data.head()
    
    def drop_navalues(self, x): #defining method for dropping NaN values in specified column 
        dropna_df = self.data[self.data[x].notna()]
        return dropna_df
    
    def data_summaries(self): #defining method to summarise counts, means, std and other data points for this data. 
        return self.data.describe()
    
    def column_average(self): #defining method to return the averages of each column. 

        column_averages = {'Period Average (days)' : [self.data["Period (days)"].mean(axis=0)],
                           'Distance Average (km)' : [self.data["Distance (km)"].mean(axis=0)],
                           'Radius Average (km)' : [self.data["Radius (km)"].mean(axis=0)],
                           'Magnitude Average' : [self.data["Magnitude"].mean(axis=0)],
                           'Eccentricity Average' : [self.data["Eccentricity"].mean(axis=0)],
                           'Inclination Degree Average (Degrees)' : [self.data["Inlination Degree"].mean(axis=0)]}
        
        averages_transposed = pd.DataFrame(column_averages).transpose() #for aesthetic purposes. 
        column_average_df = averages_transposed.rename(columns={0: "Average value"}) 
        
        display (column_average_df)
        
    def moon_info(self): #defining method to select moon and return all information for that moon. 

            x = input("Name of moon for information summary")
            
            if x in self.jupiter_data.index:
                
                moon_info = pd.DataFrame(self.jupiter_data.loc[x])
                return moon_info
            else:
                none = print("None")
                return none        

    def distributions(self): #defining method to plot histogram distributions for all columns apart from "Group"
        
        number_columns = self.data.drop("Group",  axis=1)
        for col in number_columns:
            sns.histplot(self.data, x=col, bins=5)
            plt.title(f"Distribution of {col}")
            plt.show()
   
    
    def correlation_heatmap(self): #defining method for correlation heatmap. 
        
        self.data = self.data.apply(pd.to_numeric, errors='coerce') # converting columns to numerical values and "errors='coerce" relacing invalid parsing with NaN value. 
        corr_matrix = self.data.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='viridis')  #defining method and using annot=True to display correlation values in the respective square. 
        plt.title('Correlation Heatmap')
        plt.show()
    
    def correlation_value(self): #defining method to input two variable and find correlation. 
        self.data = self.data.apply(pd.to_numeric, errors='coerce')
        corr_matrix = self.data.corr()
        
        var_1 = input("Variable 1")
        var_2 = input("Variable 2")
        
        if var_1 and var_2 in self.jupiter_data.iloc[0]: #ensuring variables are acceptable. 
            variable_correlation = print(f"The correlation value between {var_1} and {var_2} is {corr_matrix[var_1][var_2].round(3)}")
            
        else:
            none = print("None")
            return none
            return variable_correlation
        
    def plot(self, x, y): ##defining method to plot any two variables, as specified in the notebook, and see the correlation plot. 
        sns.relplot(data=self.data, x=x, y=y, hue=self.data["Group"])
        plt.title("Correlation Between Two Specified Variables")
        plt.show()
    
    
    def train(self):  #defining method to train and test the data. 
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn import linear_model
        
        
        #changing radius and period units in accordance to Keplers third law (to meters and seconds respectively)
        self.data["radius_variable"] = (self.data["Distance (km)"]*1000)**3
        self.data["period_variable"] = (self.data["Period (days)"]*86400)** 2
        
        X = self.data[["radius_variable"]]
        Y = self.data["period_variable"]
        
        model = linear_model.LinearRegression()
            
        x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size =0.3 , random_state=42)

        
        model.fit(x_train, y_train)
        
        training_pred_y = model.predict(x_train)
        testing_pred_y = model.predict(x_test)
        
        #plotting the linear regression model. 
        
        plt.figure()
        plt.scatter(X, Y)
        plt.title("Linear Regression, training and testing")
        plt.plot(x_test, testing_pred_y)
        plt.show()
        
        #printing the r2 value of the linear regression model
        print(f"The R2 score is: {r2_score(y_test,testing_pred_y)}")
        
        #using keplers third law and the gradient (model.coef_[0]) of the linear regression model to predict mass of jupiter 
        
        jupiter_mass = (4 * np.pi**2) / (model.coef_[0]*6.67e-11)
        jupiter_mass_rounded = jupiter_mass.round(3)
        
        #Calculating and Plotting the residuals of the linear regression model, by finding the difference between the observed and predicted value. 
        training_res = y_train - training_pred_y
        plt.scatter(x_train, training_res, color='red', label='Testing Residuals')
        plt.title("Plot of Residuals on the Training set")
        plt.show()
       #printing final Jupiter mass prediction. 
        return print(f"The predicted masss of Jupiter is {jupiter_mass_rounded}kg.")
    
    
 
    
    
        

      
        


        
       