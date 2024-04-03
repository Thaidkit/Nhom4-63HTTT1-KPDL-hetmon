import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import sys
from NB import NaiveBayesClassifier
from buildTree import Id3
from data_preprocrssing import preprocess_data


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

class DIABETIC_PREDICT(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Nhóm 4")
        self.geometry("1000x450")
        self.configure(bg="#F0F0F0")
        self.create_widgets()
        self.fit()

    def create_widgets(self):

        self.TUOI_label = tk.Label(self, text="Tuổi:", foreground="black")
        self.TUOI_options = {"Dưới 35": 0, "35-45": 1, "45-64": 2, "Trên 64": 3}
        self.selected_TUOI = tk.StringVar()
        self.TUOI_input = ttk.Combobox(self, values=list(self.TUOI_options.keys()), textvariable=self.selected_TUOI, width=17)

        self.GIOITINH_label = tk.Label(self, text="Giới tính:", foreground="black")
        self.GIOITINH_options = {"Nam": 1, "Nữ": 0}
        self.selected_GIOITINH = tk.StringVar()
        self.GIOITINH_input = ttk.Combobox(self, values=list(self.GIOITINH_options.keys()), textvariable=self.selected_GIOITINH, width=17)

        self.TIEUNHIEU_label = tk.Label(self, text="Đa niệu:", foreground="black")
        self.TIEUNHIEU_options = {"Không": 0, "Có": 1}
        self.selected_TIEUNHIEU = tk.StringVar()
        self.TIEUNHIEU_input = ttk.Combobox(self, values=list(self.TIEUNHIEU_options.keys()), textvariable=self.selected_TIEUNHIEU, width=17)

        self.KHATNUOC_label = tk.Label(self, text="Tình trạng khát nước quá mức:", foreground="black")
        self.KHATNUOC_options = {"Không": 0, "Có": 1}
        self.selected_KHATNUOC = tk.StringVar()
        self.KHATNUOC_input = ttk.Combobox(self, values=list(self.KHATNUOC_options.keys()), textvariable=self.selected_KHATNUOC, width=17)

        self.GIAMCAN_label = tk.Label(self, text="Tình trạng giảm cân đột ngột:", foreground="black")
        self.GIAMCAN_options = {"Không": 0, "Có": 1}
        self.selected_GIAMCAN = tk.StringVar()
        self.GIAMCAN_input = ttk.Combobox(self, values=list(self.GIAMCAN_options.keys()), textvariable=self.selected_GIAMCAN, width=17)

        self.METMOI_label = tk.Label(self, text="Tình trạng mệt mõi:", foreground="black")
        self.METMOI_options = {"Không": 0, "Có": 1}
        self.selected_METMOI = tk.StringVar()
        self.METMOI_input = ttk.Combobox(self, values=list(self.METMOI_options.keys()), textvariable=self.selected_METMOI, width=17)

        self.THEMAN_label = tk.Label(self, text="Tình trạng thèm ăn:", foreground="black")
        self.THEMAN_options = {"Không": 0, "Có": 1}
        self.selected_THEMAN = tk.StringVar()
        self.THEMAN_input = ttk.Combobox(self, values=list(self.THEMAN_options.keys()), textvariable=self.selected_THEMAN, width=17)

        self.NAMPK_label = tk.Label(self, text="Bị nấm phụ khoa:", foreground="black")
        self.NAMPK_options = {"Không": 0, "Có": 1}
        self.selected_NAMPK = tk.StringVar()
        self.NAMPK_input = ttk.Combobox(self, values=list(self.NAMPK_options.keys()), textvariable=self.selected_NAMPK, width=17)

        self.MATMO_label = tk.Label(self, text="Tình trạng mắt bị mờ:", foreground="black")
        self.MATMO_options = {"Không": 0, "Có": 1}
        self.selected_MATMO = tk.StringVar()
        self.MATMO_input = ttk.Combobox(self, values=list(self.MATMO_options.keys()), textvariable=self.selected_MATMO, width=17)

        self.NGUA_label = tk.Label(self, text="Tình trạng bị ngứa ở một số bộ phận:", foreground="black")
        self.NGUA_options = {"Không": 0, "Có": 1}
        self.selected_NGUA = tk.StringVar()
        self.NGUA_input = ttk.Combobox(self, values=list(self.NGUA_options.keys()), textvariable=self.selected_NGUA, width=17)

        self.KHOCHIU_label = tk.Label(self, text="Tình trạng khó chịu, dễ cáu giận:", foreground="black")
        self.KHOCHIU_options = {"Không": 0, "Có": 1}
        self.selected_KHOCHIU = tk.StringVar()
        self.KHOCHIU_input = ttk.Combobox(self, values=list(self.KHOCHIU_options.keys()), textvariable=self.selected_KHOCHIU, width=17)

        self.VETTHUONG_label = tk.Label(self, text="Tình trạng vết thương lâu lành:", foreground="black")
        self.VETTHUONG_options = {"Không": 0, "Có": 1}
        self.selected_VETTHUONG = tk.StringVar()
        self.VETTHUONG_input = ttk.Combobox(self, values=list(self.VETTHUONG_options.keys()), textvariable=self.selected_VETTHUONG, width=17)

        self.CHUCNANGCO_label = tk.Label(self, text="Suy giảm chức năng một phần của cơ:", foreground="black")
        self.CHUCNANGCO_options = {"Không": 0, "Có": 1}
        self.selected_CHUCNANGCO = tk.StringVar()
        self.CHUCNANGCO_input = ttk.Combobox(self, values=list(self.CHUCNANGCO_options.keys()), textvariable=self.selected_CHUCNANGCO, width=17)

        self.CANGCO_label = tk.Label(self, text="Tình trạng căng cơ:", foreground="black")
        self.CANGCO_options = {"Không": 0, "Có": 1}
        self.selected_CANGCO = tk.StringVar()
        self.CANGCO_input = ttk.Combobox(self, values=list(self.CANGCO_options.keys()), textvariable=self.selected_CANGCO, width=17)

        self.RUNGTOC_label = tk.Label(self, text="Tình trạng rụng nhiều tóc:", foreground="black")
        self.RUNGTOC_options = {"Không": 0, "Có": 1}
        self.selected_RUNGTOC = tk.StringVar()
        self.RUNGTOC_input = ttk.Combobox(self, values=list(self.RUNGTOC_options.keys()), textvariable=self.selected_RUNGTOC, width=17)

        self.BEOPHI_label = tk.Label(self, text="Tình trạng béo phì hoặc có tiền sử bệnh:", foreground="black")
        self.BEOPHI_options = {"Không": 0, "Có": 1}
        self.selected_BEOPHI = tk.StringVar()
        self.BEOPHI_input = ttk.Combobox(self, values=list(self.BEOPHI_options.keys()), textvariable=self.selected_BEOPHI, width=17)

        labels = []
        inputs = []

        labels.append(self.TUOI_label)
        inputs.append(self.TUOI_input)

        labels.append(self.MATMO_label)
        inputs.append(self.MATMO_input)

        labels.append(self.GIOITINH_label)
        inputs.append(self.GIOITINH_input)

        labels.append(self.NGUA_label)
        inputs.append(self.NGUA_input)

        labels.append(self.TIEUNHIEU_label)
        inputs.append(self.TIEUNHIEU_input)

        labels.append(self.KHOCHIU_label)
        inputs.append(self.KHOCHIU_input)

        labels.append(self.KHATNUOC_label)
        inputs.append(self.KHATNUOC_input)

        labels.append(self.VETTHUONG_label)
        inputs.append(self.VETTHUONG_input)

        labels.append(self.GIAMCAN_label)
        inputs.append(self.GIAMCAN_input)

        labels.append(self.CHUCNANGCO_label)
        inputs.append(self.CHUCNANGCO_input)

        labels.append(self.METMOI_label)
        inputs.append(self.METMOI_input)

        labels.append(self.CANGCO_label)
        inputs.append(self.CANGCO_input)

        labels.append(self.THEMAN_label)
        inputs.append(self.THEMAN_input)

        labels.append(self.RUNGTOC_label)
        inputs.append(self.RUNGTOC_input)

        labels.append(self.NAMPK_label)
        inputs.append(self.NAMPK_input)

        labels.append(self.BEOPHI_label)
        inputs.append(self.BEOPHI_input)

        # Arrange labels and comboboxes in 2 columns
        for i, (label, input_box) in enumerate(zip(labels, inputs)):
            if i % 2 == 0:
                column_index = 0
            else:
                column_index = 1
            row_index = i // 2
            label.grid(row=row_index, column=column_index * 2, sticky=tk.E, padx=10, pady=5)
            input_box.grid(row=row_index, column=column_index * 2 + 1, padx=10, pady=5)

        self.prediction_button = tk.Button(self, text="Dự đoán", command=self.perform_prediction, foreground="black")
        self.prediction_button.grid(row=len(labels)//2 + 1, columnspan=4, padx=10, pady=10, sticky=tk.W)

        self.prediction_label = tk.Label(self, text="Kết quả ID3: (Có : 1; Không : 0) ", foreground="black")
        self.prediction_label.grid(row=len(labels)//2 + 2, columnspan=4, padx=10, pady=10, sticky=tk.W)

        self.prediction_labelNavie = tk.Label(self, text="Kết quả Navie: (Có : 1; Không : 0) ", foreground="black")
        self.prediction_labelNavie.grid(row=len(labels)//2 + 3, columnspan=4, padx=10, pady=10, sticky=tk.W)

        self.reset_button = tk.Button(self, text="Reset", command=self.reset_values, foreground="black")
        self.reset_button.grid(row=len(labels)//2 + 4, columnspan=4, padx=10, pady=10, sticky=tk.W)

    def fit_NaiveBayes(self, X_NB, y_NB, feature_names):
        model2 = NaiveBayesClassifier()
        model2.fit(X_NB, y_NB)
        model2.feature_names = feature_names 
        return model2

    def fit(self):
        data = pd.read_csv('/Users/mac/Documents/Đại học/Nam3/Ki2_Nam3/Khai phá dữ liệu/Báo cáo/Nhom4_63HTTT1/Code/Data/Data/DaXuLy.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        feature_names = X.columns.tolist()  # Danh sách tên của các tính năng
        self.NB = self.fit_NaiveBayes(X_train, y_train, feature_names)


    def perform_prediction(self):
        mapping = {
            "Dưới 35": 0, "35-45": 1, "45-64": 2, "Trên 64": 3,
            "Nam": 1, "Nữ": 0,
            "Không": 0, "Có": 1
        }

        TUOI_temp = mapping[self.selected_TUOI.get()]
        GIOITINH_temp = mapping[self.selected_GIOITINH.get()]
        TIEUNHIEU_temp = mapping[self.selected_TIEUNHIEU.get()]
        KHATNUOC_temp = mapping[self.selected_KHATNUOC.get()]
        GIAMCAN_temp = mapping[self.selected_GIAMCAN.get()]
        METMOI_temp = mapping[self.selected_METMOI.get()]
        THEMAN_temp = mapping[self.selected_THEMAN.get()]
        NAMPK_temp = mapping[self.selected_NAMPK.get()]
        MATMO_temp = mapping[self.selected_MATMO.get()]
        NGUA_temp = mapping[self.selected_NGUA.get()]
        KHOCHIU_temp = mapping[self.selected_KHOCHIU.get()]
        VETTHUONG_temp = mapping[self.selected_VETTHUONG.get()]
        CHUCNANGCO_temp = mapping[self.selected_CHUCNANGCO.get()]
        CANGCO_temp = mapping[self.selected_CANGCO.get()]
        RUNGTOC_temp = mapping[self.selected_RUNGTOC.get()]
        BEOPHI_temp = mapping[self.selected_BEOPHI.get()]

        attribute = [TUOI_temp , GIOITINH_temp, TIEUNHIEU_temp, KHATNUOC_temp, GIAMCAN_temp, METMOI_temp, THEMAN_temp, NAMPK_temp, MATMO_temp, NGUA_temp, KHOCHIU_temp, VETTHUONG_temp, CHUCNANGCO_temp, CANGCO_temp, RUNGTOC_temp, BEOPHI_temp]

        # Predict using Decision Tree model
        column_names = ['TUOI', 'GIOITINH', 'TIEUNHIEU', 'KHATNUOC', 'GIAMCAN', 'METMOI', 'THEMAN', 'NAMPK', 'MATMO', 'NGUA', 'KHOCHIU', 'VETTHUONG', 'CHUCNANGCO', 'CANGCO', 'RUNGTOC', 'BEOPHI']
        df_attribute = pd.DataFrame.from_records([attribute], columns=column_names)

        # print(df_attribute)

         # Sử dụng cây quyết định để dự đoán
        predicted_value_DT = Id3.predict(tree, df_attribute)

        # Predict using Naive Bayes model
        predicted_value_NB = self.NB.predict(np.array([attribute]))

        # Display the prediction result
        #self.prediction_label.config(text=f"Kết quả dự đoán của ID3: {predicted_value_DT[0]}\nKết quả dự đoán của Naive Bayes: {predicted_value_NB[0]}")

        # Hiển thị kết quả dự đoán lên giao diện
        self.prediction_label.config(text="Kết quả ID3: {}".format(predicted_value_DT[0]))

        self.prediction_labelNavie.config(text="Kết quả Navie: {}".format(predicted_value_NB[0]))

    def reset_values(self):
        # Đặt giá trị ban đầu cho tất cả các comboboxes
        self.selected_TUOI.set("")
        self.selected_GIOITINH.set("")
        self.selected_TIEUNHIEU.set("")
        self.selected_KHATNUOC.set("")
        self.selected_GIAMCAN.set("")
        self.selected_METMOI.set("")
        self.selected_THEMAN.set("")
        self.selected_NAMPK.set("")
        self.selected_MATMO.set("")
        self.selected_NGUA.set("")
        self.selected_KHOCHIU.set("")
        self.selected_VETTHUONG.set("")
        self.selected_CHUCNANGCO.set("")
        self.selected_CANGCO.set("")
        self.selected_RUNGTOC.set("")
        self.selected_BEOPHI.set("")

        # Đặt giá trị kết quả dự đoán về trạng thái ban đầu
        self.prediction_label.configure(text="Kết quả: (Có : 1; Không : 0)")
        self.prediction_labelNavie.config(text="Kết quả Navie: (Có : 1; Không : 0)")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data("//Users/mac/Documents/Đại học/Nam3/Ki2_Nam3/Khai phá dữ liệu/Báo cáo/Nhom4_63HTTT1/Code/Code/DaXuLy.csv")
    tree = Id3.build_tree(X_train, y_train, alpha=0)
    app = DIABETIC_PREDICT()
    app.mainloop()
