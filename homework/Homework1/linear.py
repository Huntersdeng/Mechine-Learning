import numpy as np
import xlrd

#load the dataset
DATA_FILE="dataset.xlsx"
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
X = np.zeros(shape=(8,17))
Y = np.zeros(shape=(17,))
for i in range(22,39):
    for j in range(15,23):
        X[j - 15][i - 22] = sheet.cell_value(i,j)
for i in range(22,39):
    Y[i - 22] = sheet.cell_value(i,23)
X_train = X[:,0:14]
Y_train = Y[0:14]
X_test = X[:,14:17]
Y_test = Y[14:17]