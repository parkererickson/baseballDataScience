from pybaseball import statcast
import pandas as pd
import numpy as np
from pybaseball import statcast_pitcher
from pybaseball import playerid_lookup

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from keras.utils import to_categorical

from tensorflow import feature_column
from tensorflow.keras import layers
import tensorflow as tf

pid = playerid_lookup('berrios','jose')["key_mlbam"][0]
print(pid)
# get all available data
data = statcast_pitcher('2017-03-01', '2019-10-10', player_id = pid)

data = data[["pitch_type", "bat_score", "fld_score", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "inning_topbot", "pitch_number", "p_throws", "balls", "strikes", "stand", "batter", "release_speed", "description"]]

data = data[data.pitch_type != 'EP']
data = data[data.pitch_type != 'PO']

data[["on_3b", "on_2b", "on_1b"]] = data[["on_3b", "on_2b", "on_1b"]].replace(np.nan, 0)
data.loc[data.on_3b > 0, "on_3b"] = 1
data.loc[data.on_2b > 0, "on_2b"] = 1
data.loc[data.on_1b > 0, "on_1b"] = 1
data = data.dropna()

data["prev_pitch"] = data["pitch_type"].shift(-1)
data["prev_batter"] = data["batter"].shift(-1)
data["release_speed"] = data["release_speed"].shift(-1)
data["description"] = data["description"].shift(-1)

def filterAB(row):
    if row["batter"] == row["prev_batter"]:
        return row["prev_pitch"]
    else:
        return "NB"
    
def filterSpeed(row):
    if row["batter"] == row["prev_batter"]:
        return row["release_speed"]
    else:
        return 0
    
def filterDesc(row):
    if row["batter"] == row["prev_batter"]:
        return row["description"]
    else:
        return "NB"

data["prev_pitch"] = data.apply(lambda row: filterAB(row), axis = 1)
data["release_speed"] = data.apply(lambda row: filterSpeed(row), axis=1)
data["description"] = data.apply(lambda row: filterDesc(row), axis=1)

data = data.drop("prev_batter", 1)
data = data.drop("batter", 1)

data = pd.concat([data,pd.get_dummies(data["prev_pitch"], prefix='prev_pitch')],axis=1)
data = pd.concat([data,pd.get_dummies(data["p_throws"], prefix='pitcher_throws')],axis=1)
data = pd.concat([data,pd.get_dummies(data["stand"], prefix='batter_stands')],axis=1)
data = pd.concat([data,pd.get_dummies(data["description"], prefix='prev_desc')],axis=1)
data = data.drop(['inning_topbot', 'p_throws', 'stand', "description", "prev_pitch"], axis = 1)

categoricalMask = data.dtypes==object
categoricalCols = data.columns[categoricalMask].tolist()
categoricalCols



le = LabelEncoder()
data[categoricalCols] = data[categoricalCols].apply(lambda col: le.fit_transform(col))
data = data.sample(frac=1).reset_index(drop=True)

features = data.columns.tolist()[1:]
X = data[features]
y = data["pitch_type"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

print(y_train.value_counts())
smote = SMOTE("all")
X_train, y_train = smote.fit_sample(X_train, y_train)
print(y_train.value_counts())

num_classes=len(data["pitch_type"].unique())
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(len(features),), activation='relu', name='fc1'), #layer 1
    tf.keras.layers.Dense(8, activation='relu', name='fc2'), #layer 2
    tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
  ])

  model.compile(optimizer='adam', 
                loss='binary_crossentropy',
                metrics=['accuracy'])
  return model

model = get_compiled_model()
model.fit(X_train, y_train, batch_size=100, epochs=200)

results = model.evaluate(X_test, y_test)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

pitches = le.inverse_transform(data["pitch_type"].unique())

data["pitch_type"].unique()

some_data = data.sample(n=1)
ynew = model.predict_classes(some_data[features])
print(some_data)
print(le.inverse_transform(ynew))