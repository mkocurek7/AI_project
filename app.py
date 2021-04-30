import np as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState
from matplotlib import pyplot
import seaborn as sns
from pandas import read_csv
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix
from seaborn import scatterplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import Imputer
from numpy import nan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import warnings
warnings.filterwarnings(action='ignore')

#raw_data= read_csv('train.csv') #csv ze wszystkimi rekordami 
raw_data=read_csv('train2.csv') #csv połączone test.csv+train.csv(bez rekordow gdzie w calym wierszu brakuje danych)
dataset=raw_data.copy()

#print(dataset.shape) #shape zwraca (ilosc wierszy, ilosc kolumn)
#print(dataset.isna().sum()) #zwraca ilosc danych ktore sa nieprawidlowe

#łatwe sprawdzenie zmiennych kategorycznych jakie sa
table=pd.pivot_table(dataset, values='quantity', index=['city', 'shop', 'brand'], columns='container',  aggfunc=np.sum)
#print(table)

#jesli miałby sie czepiac ze dane lekko zmodyfikowane a nie surowe z kaggle:
#pobranie danych z train.csv i tets.csv-> TODO:
#dataset = dataset.dropna() #usuwa nieprawidlowe wiersze z danymi (usuwa takze wiersz gdzie 1 dana jest nieprawidlowa)
#usunac jak najmniejsza ilosc rekordow-> reszte ladnie zastąpic-> jesli sie da , do przemyslenia
#no i jesli sie da połączyc train i test i jakos zczytac


#organizacja zmiennych jakosciowych-> zamiast nazw sa liczby
city_map={'Athens':1,'Irakleion':2,'Larisa':3,'Patra':4, 'Thessaloniki':5 }
shop_map={'shop_1':1, 'shop_2':2, 'shop_3':3,'shop_4':4, 'shop_5':5,'shop_6':6}
brand_map={'kinder-cola':1, 'adult-cola':2, 'orange-power':3, 'gazoza':4,'lemon-boost':5}
container_map={'can':1, 'glass':2,'plastic':3}
container_volume_map={'1.5lt':1,'330ml':2,'500ml':3}
dataset['city']=dataset['city'].map(city_map)
dataset['shop']=dataset['shop'].map(shop_map)
dataset['brand']=dataset['brand'].map(brand_map)
dataset['container']=dataset['container'].map(container_map)
dataset['capacity']=dataset['capacity'].map(container_volume_map)

#brakujace dane i uzupelnianie
print(dataset.isnull().sum()) # do sprawdzenia gdzie brakuje danych

#jakby byl problem z nullami to odkomentowac konwersje na NaN
#dataset[['lat','long','container','capacity']]=dataset[['lat','long','container','capacity',]].replace(0,np.nan)
#dataset['capacity']=dataset['capacity'].replace(0,np.nan)

#wypelnienie brkaujacych danych
dataset.fillna(dataset.mean(), inplace=True)
dataset.fillna(dataset['capacity'].mode()[0], inplace=True)
print(dataset.isnull().sum())

#wypisanie linijek z brakujacymi danymi jesli takie sie pojawią
#null_data = dataset[dataset.isnull().any(axis=1)]
#print(null_data)

print(dataset.describe()) #opis statystyczny danych :)

#DIAGRAMY
#print("")
##wykresy wasy-box
#dataset.columns
#print('pop')
#plt.boxplot(dataset['pop'])
#plt.show()
#print('price')
#plt.boxplot(dataset['price'])
#plt.show()
#print('quantity')
#plt.boxplot(dataset['quantity'])
#plt.show()
#
##histogramy (lat-> latitude , long-> longitude)
#dataset.hist('pop')
#plt.show()
#dataset.hist('price')
#plt.show()
#dataset.hist('quantity')
#plt.show()
#dataset.hist('city')
#plt.show()
#dataset.hist('brand')
#plt.show()
#dataset.hist('container')
#plt.show()
##przy container,brand, city konkretne wartosci typu 1 ,2 oznaczaja miasta/rodzaj opakowania/marke
#
#sns.relplot(x=dataset['price'], y=dataset['container'])
#sns.relplot(x=dataset['price'], y=dataset['brand'])
#sns.relplot(x=dataset['price'], y=dataset['quantity']) #zaleznosc cena ilosc
#sns.relplot(x=dataset['city'], y=dataset['pop'])
#sns.relplot(x=dataset['city'], y=dataset['quantity'])

# quantity jest zmienna wyjsciowa wiec wywalamy ja ze zbioru x i wrzucamy do zbioru y
#lat i long nie wydaja sie potrzebne do czegokolwiek
# axis=1 powoduje ze indeksy nie beda uwzgldniane podczas trenowania

#X = dataset.drop("quantity", axis=1).copy()
#y = dataset["quantity"].copy()
#x_train, x_test, y_train, y_test = ms.train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

dataset = dataset.astype({"pop": "int"})
dataset = dataset.astype({"price": "float64"})
dataset = dataset.astype({"quantity": "int"})
dataset = dataset.astype({"capacity": "int"})
dataset = dataset.astype({"city": "int"})
dataset = dataset.astype({"brand": "int"})
dataset = dataset.astype({"container": "int"})
dataset = dataset.astype({"shop": "int"})

dataset.pop("lat")
dataset.pop("long")
dataset.pop("date")

dataset.info()

x = dataset.drop('quantity', axis=1).copy()
y = dataset['quantity'].copy()

y-=1
#print(y)

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=False)
y_train.hist()
plt.ylabel('Y_TRAIN')
plt.show()

#alternatywa
#train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=None, shuffle=False)
#x_train = train_dataset.copy()
#x_test = test_dataset.copy()
#y_train = x_train.pop('quantity')
#y_test = x_test.pop('quantity')
#normalizer = preprocessing.Normalization()
#normalizer.adapt(np.array(x_train))
#print(normalizer.mean.numpy())
#print(y_test)

model = keras.models.Sequential([
    # definicja pierwszej wartwy - argumentem jest liczba zmiennych wejsciowych
    keras.layers.Input(shape=x_train.shape[1]),
    # definicja warstw ukrytych - pierwszy argument to ilosc neuronow na warstwie, drugi to funkcja aktywacji
    # z definicjami warstw ukrytych powinno sie eksperymentowac poprzez np zmiane liczby neuronow na warstwie
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    # pierszy argument - liczba neuronow wyjsciowych
    keras.layers.Dense(400000, activation="softmax")
])

model.compile(
    # funkcja straty
    loss="sparse_categorical_crossentropy",
    # algorytm wyznaczania wag
    optimizer="adam",
    # metryki procesu uczenia
    metrics=["accuracy"]
)
model.summary()

#model_history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

#plt.plot(model_history.history["val_accuracy"])
#np.max(model_history.history["accuracy"])
