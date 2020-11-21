# %%

# Подключаем зависимости
import matplotlib.pyplot as pt
import pandas as pd

#Указываем путь до файла с данными
csv_file_path = './archive/emnist-letters-train.csv'

# %%
# Отбираем только нужные нам буквы (A,B,C,D,E,F,H,J,L,K)
df = pd.read_csv(csv_file_path)
data = df[df.iloc[:, 0].isin([1,2,3,4,5,6,8,10,11,12])]
#%%
#Полученные данные
print(data)

# %%
#Случайная буква из набора
d = data.iloc[11, 1:].to_numpy()
d.shape = (28, 28)
pt.imshow(255 - d, cmap='gray')
pt.show()
