
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


class ModeloOraculo:

    def __init__(self):
        self.modelo = RandomForestClassifier(random_state=42)
        self.entrenado = False

    def entrenar(self,datos):
        X = datos[['temperatura','niebla','hora']] #es una matriz, por eso en mayuscula
        y = datos['perdido'] #es un vector (un unico dato) -> minusculas


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.modelo.fit(X_train,y_train)
        self.entrenado = True

        y_pred = self.modelo.predict(X_test)

        precision =accuracy_score(y_test,y_pred)
        matriz = confusion_matrix(y_test,y_pred)

        return {
            'precision': precision,
            'confusion_matrix': matriz
        }

    def predecir(self, temperatura, niebla, hora):
        if not self.entrenado:
            raise Exception("¡El oráculo aún no ha sido entrenado, Señor Frodo!")
        
        entrada = pd.DataFrame({
        'temperatura': [temperatura],
        'niebla': [niebla],
        'hora': [hora]
        })
        prediccion = self.modelo.predict(entrada)

        return prediccion[0]

