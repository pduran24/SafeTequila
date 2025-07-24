

import numpy as np
import pandas as pd

class CrearDatos:

    def __init__(self, num_ejemplos=100, seed=42):
        self.num_ejemplos = num_ejemplos
        self.seed = seed
    

    def generar (self):
        np.random.seed(self.seed)
        temperatura = np.random.normal(15,10,self.num_ejemplos)
        niebla = np.random.choice([0,1],self.num_ejemplos)
        hora = np.random.randint(0,24,self.num_ejemplos)
        perdido = ((temperatura < 10) & (niebla == 1) & ((hora < 6) | (hora > 20))).astype(int)


        datos = pd.DataFrame({
            'temperatura': temperatura,
            'niebla': niebla,
            'hora': hora,
            'perdido': perdido
        })

        return datos


    






