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
dataset['city']=dataset['city'].map(city_map)
dataset['shop']=dataset['shop'].map(shop_map)
dataset['brand']=dataset['brand'].map(brand_map)
dataset['container']=dataset['container'].map(container_map)

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
print("")
#wykresy wasy-box 
dataset.columns
print('pop')
plt.boxplot(dataset['pop'])
plt.show()
print('price')
plt.boxplot(dataset['price'])
plt.show()
print('quantity')
plt.boxplot(dataset['quantity'])
plt.show()

#histogramy (lat-> latitude , long-> longitude)
dataset.hist('pop')
plt.show()
dataset.hist('price')
plt.show()
dataset.hist('quantity')
plt.show()
dataset.hist('city')
plt.show()
dataset.hist('brand')
plt.show()
dataset.hist('container')
plt.show()
#przy container,brand, city konkretne wartosci typu 1 ,2 oznaczaja miasta/rodzaj opakowania/marke

sns.relplot(x=dataset['price'], y=dataset['container'])
sns.relplot(x=dataset['price'], y=dataset['brand'])
sns.relplot(x=dataset['price'], y=dataset['quantity']) #zaleznosc cena ilosc 
sns.relplot(x=dataset['city'], y=dataset['pop']) 
sns.relplot(x=dataset['city'], y=dataset['quantity']) 

#podzial na train i test
#train_dataset= dataset.sample(frac=0.8, random_state=0)
#test_dataset=dataset.drop(train_dataset.index)

# wstepnie do wywalenia
# rng = RandomState()
# train_dataset = raw_data.sample(frac=0.8, random_state=rng)
# test_dataset = raw_data.sample[~dataset.index.isin(train_dataset.index)]

# wstepnie do wywalenia
# train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, test_size=0.2, random_state=None, shuffle=False)
# print(train_dataset.shape)
# print(test_dataset.shape)

# quantity jest zmienna wyjsciowa wiec wywalamy ja ze zbioru x i wrzucamy do zbioru y
# axis=1 powoduje ze indeksy nie beda uwzgldniane podczas trenowania

# nie wiem czy daty tez nie wyrzucic z x
x = dataset.drop(["quantity", "date"], axis=1)
y = dataset["quantity"]

dataset["quantity"].hist()

y -= 1

print(dataset["quantity"].value_counts())

#opcjonalnie - zaokraglenie danych w kolumnie quatity tak by bylo mniej przedzialow  TODO:

x_train, x_test, y_train, y_test = ms.train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=None, shuffle=False) #, stratify=y

model = keras.models.Sequential([
    # definicja pierwszej wartwy - argumentem jest liczba zmiennych wejsciowych
    keras.layers.Input(shape=x_train.shape[1]),
    # definicja warstw ukrytych - pierwszy argument to ilosc neuronow na warstwie, drugi to funkcja aktywacji
    # z definicjami warstw ukrytych powinno sie eksperymentowac poprzez np zmiane liczby neuronow na warstwie
    keras.layers.Dense(500, activation="relu"),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(400, activation="relu"),
    keras.layers.Dense(200, activation="relu"),
    keras.layers.Dense(150, activation="relu"),
    # definicja warstwy wejsciowej
    # pierszy argument - liczba neuronow wyjsciowych (prawdopodobnie powinny byc 3)
    keras.layers.Dense(3, activation="softmax")
])

model.compile(
    # funkcja straty
    loss="sparse_categorical_crossentropy",
    # algorytm wyznaczania wag
    optimizer="adam",
    # metryki procesu uczenia
    metrics=["accuracy"]
)


print(x.value_counts())
model.summary()
#zmapowac opakowania o danej pojemnosci do intow
x_train=np.asarray(x_train).astype(np.int)
y_train=np.asarray(y_train).astype(np.int)
model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test))



