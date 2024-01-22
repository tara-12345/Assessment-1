import pandas as pd 

class Moons:
    def __init__(self):
        self.data = self.load_data() #attributing data set to moons data 
        self.names = self.load_names()
        
    def load_data (self): #opening and reading moons database 
        
        connectable = f"sqlite:///jupiter.db"
        query = "SELECT * FROM moons"
        
        jupiter_data = pd.read_sql(query, connectable)
        jupiter_data.set_index("moon", inplace=True)
        
        return jupiter_data

    def load_names(self): #determing names for moons 
        names_list = []
        for names in self.load_data().index:
            names_list.append(names)
            
        return names_list
            
    def plot(self, x, y):
        import matplotlib.pyplot as plt
        plt.scatter(self.data[x], self.data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        
        plt.show()