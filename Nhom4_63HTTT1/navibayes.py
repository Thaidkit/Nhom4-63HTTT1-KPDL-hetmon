import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}
        self.feature_probs = {}

    def fit(self, X, y):
        # Convert DataFrame to numpy array
        X = X.to_numpy()
        
        # Tính xác suất của lớp
        total_samples = len(y)
        unique_labels, label_counts = np.unique(y, return_counts=True)

        for label, count in zip(unique_labels, label_counts):
            self.class_probs[label] = count / total_samples

        # Tính xác suất đặc trưng
        num_features = X.shape[1]
        for label in set(y):
            Label_index = [i for i in range(total_samples) if y.iloc[i] == label]
            label_mau = X[Label_index]

            self.feature_probs[label] = []
            for feature in range(num_features):
                feature_values = label_mau[:, feature]
                unique_feature_values, value_counts = np.unique(feature_values, return_counts=True)
                feature_prob = {value: count / len(label_mau) for value, count in zip(unique_feature_values, value_counts)}
                self.feature_probs[label].append(feature_prob)

    def fit_NaiveBayes(self, X_NB, y_NB, feature_names):
        model2 = NaiveBayesClassifier()
        model2.fit(X_NB, y_NB)
        model2.feature_names = feature_names 
        return model2

    def fitNB(self, X, y):
        feature_names = X.columns.tolist()  # Danh sách tên của các tính năng
        self.NB = self.fit_NaiveBayes(X, y, feature_names)
        return self.NB
    
    # Hàm tính giá trị dự đoán
    def predict(self, X):
        xacsuat = []
        for mau in X:
            probs = {}
            for label in self.class_probs:
                class_prob = self.class_probs[label]
                feature_probs = self.feature_probs[label]

                for i, value in enumerate(mau):
                    if value in feature_probs[i]:
                        class_prob *= feature_probs[i][value]
                    else:
                        class_prob = 0  # Nếu không thấy giá trị nào trong quá trình huấn luyện, xác suất sẽ là 0

                probs[label] = class_prob

            xacsuat.append(max(probs, key=probs.get))

        return xacsuat
