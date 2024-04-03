import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import sys
import matplotlib.pyplot as plt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

class NaiveBayes(tk.Tk):
    
    def __init__(self):
        self.classes = None
        self.class_probabilities = None
        self.feature_probabilities = None
        self.fit()


    def fit(self):
        # Bước 1: Đọc dữ liệu từ file csv
        data = pd.read_csv("./DaXuLy.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # Chia dữ liệu thành tập train và tập test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        # Bước 2: Tính xác suất của mỗi phân lớp
        class_counts = y_train.value_counts()  #số mẫu thuộc 2 lopws0,1
        self.classes = class_counts.index  #lưu các nhãn lớp index [0,1]
        self.class_probabilities = class_counts / len(y_train)    
        # Bước 3: Tính xác suất của từng thuộc tính trên từng phân lớp
        self.feature_probabilities = {}
        for feature_column in X_train.columns:
            feature_probabilities_given_class = {}# xác suất của từng thuộc tính/lớp
            for class_label in self.classes:
                subset = X_train[y_train == class_label]   # subnet chứa các mẫu chỉ thuộc lớp đó
                feature_counts = subset[feature_column].value_counts() #đếm gia trị xuất hiện trong  nhãn này
                probabilities = feature_counts / len(subset)# xác suất trong từng lớp
                feature_probabilities_given_class[class_label] = probabilities  #lưu lại xác suất của nhãn 0 trc 1 sau
            self.feature_probabilities[feature_column] = feature_probabilities_given_class   #lưu lại xs của từng giá trị trên từng lớp
            
    def predict(self, X_test):
        predictions = []
        for _,sample in X_test.iterrows(): #lấy chỉ mục hàng và dữ liệu hàng samples
            probabilities = {}
            for class_label in self.classes:
                class_probability = self.class_probabilities[class_label] #xác suất của từng nhãn
                feature_probabilities = self.feature_probabilities    # xác suất của từng thuộc tính /lớp
                for feature_column, feature_value in sample.items():#kiểm tra xem tôn tại giá trị thuộc tính trong samples k
                    if feature_column not in feature_probabilities:  #nếu không tồn tại cột thì bỏ qua
                        continue
                    #nếu giá trị của cột đó không tồn tại trong từ điển thì xs = 0
                    if feature_value not in feature_probabilities[feature_column][class_label]:
                        probability = 0
                    else:
                        probability = feature_probabilities[feature_column][class_label][feature_value]
                    class_probability *= probability

                probabilities[class_label] = class_probability

            predicted_class = max(probabilities, key=probabilities.get)
            predictions.append(predicted_class)

        return predictions         
                


if __name__ == "__main__":
    app = NaiveBayes()
    app.mainloop()