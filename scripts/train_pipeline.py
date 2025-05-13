import pandas as pd
from modeling.train_model import train_model

# حمّل الداتا الجديدة
df = pd.read_csv("data/raw/cardio_data.csv", delimiter=",")

# نفّذ التدريب وحفظ الموديل تلقائيًا
train_model(df)
