import numpy as np
import pandas as pd
import math
import graphviz

class Id3:
    def info(p, n):
        if p == 0 or n == 0:
            return 0
        else:
            return (-p/(p+n))*math.log2(p/(p+n)) + (-n/(p+n))*math.log2(n/(p+n))

    def entropy(attribute_values, target_values):
        total = len(target_values)
        attribute_value_counts = pd.Series(attribute_values).value_counts()
        attribute_entropy = 0
        for attribute_value, count in attribute_value_counts.items():
            target_subset = target_values[attribute_values == attribute_value]
            p = len(target_subset[target_subset == 1])
            n = len(target_subset) - p
            attribute_entropy += (count/total) * Id3.info(p, n)
        return attribute_entropy

    def gain(attribute_values, target_values):
        p = len(target_values[target_values == 1])
        n = len(target_values) - p
        
        parent_entropy = Id3.info(p, n)
        attribute_entropy = Id3.entropy(attribute_values, target_values)

        return parent_entropy - attribute_entropy

    class Node:
        def __init__(self, attribute=None, value=None, label=None):
            self.attribute = attribute
            self.value = value
            self.label = label
            self.children = {}

        def add_child(self, value, node):
            self.children[value] = node

    def build_tree(X_train, y_train, alpha=0):
        # Lấy ra danh sách các thuộc tính trong X_train
        attributes = X_train.columns.tolist()

        # Kiểm tra xem còn thuộc tính nào để phân chia cây hay không
        if len(attributes) == 0:
            label = y_train.value_counts().idxmax()
            return Id3.Node(label=label)

        # Tính toán số lượng mẫu thuộc lớp dương và lớp âm trong tập huấn luyện
        p = len(y_train[y_train == 1])
        n = len(y_train) - p

        # Kiểm tra điều kiện dừng: nếu không còn mẫu thuộc một lớp hoặc alpha đạt đến ngưỡng thì trả về node lá
        if p == 0 or n == 0 or alpha >= 1:
            label = y_train.value_counts().idxmax()
            return Id3.Node(label=label)

        # Tìm thuộc tính có gain cao nhất
        information_gains = []
        for attribute in attributes:
            information_gain = Id3.gain(X_train[attribute], y_train)
            information_gains.append(information_gain)
        best_attribute_idx = np.argmax(information_gains)
        best_attribute = attributes[best_attribute_idx]

        # Tạo một node mới với thuộc tính tốt nhất
        node = Id3.Node(attribute=best_attribute)

        # Xóa thuộc tính đã chọn để phân chia khỏi danh sách các thuộc tính
        attributes.remove(best_attribute)

        # Xây dựng các nhánh con cho node hiện tại
        attribute_values = X_train[best_attribute].unique()
        for attribute_value in attribute_values:
            subset_indices = X_train[best_attribute] == attribute_value
            subset_X_train = X_train.loc[subset_indices].drop(columns=best_attribute)
            subset_y_train = y_train.loc[subset_indices]
            child_node = Id3.build_tree(subset_X_train, subset_y_train, alpha)
            node.add_child(attribute_value, child_node)

        # Kiểm tra xem liệu cây con có thể được cắt tỉa hay không
        if alpha > 0:
            subtree_labels = [child_node.label for child_node in node.children.values()]
            print(subtree_labels)
            subtree_p = subtree_labels.count(1)
            subtree_n = len(subtree_labels) - subtree_p
            subtree_error = (subtree_p + alpha) / (subtree_p + subtree_n + 2 * alpha)
            if (p + n) * subtree_error <= p + alpha or (p + n) * subtree_error <= n + alpha:
                node.children = {}

        return node
    
    def predict(node, X_test):
        predict_labels = []
        for index, instance in X_test.iterrows():
            current_node = node
            while current_node.label is None:
                attribute = current_node.attribute
                attribute_value = instance[attribute]
                if attribute_value not in current_node.children:
                     # Nếu giá trị thuộc tính không được tìm thấy trong các nhánh, dự đoán sai
                    predict_labels.append(2)
                    break
                current_node = current_node.children[attribute_value]
            else:
                # Nếu cây quyết định đã đạt đến một nút lá, dự đoán nhãn của nút lá đó
                predict_labels.append(current_node.label)
        return predict_labels

    def visualize_tree(node, dot=None):
        if dot is None:
            dot = graphviz.Digraph()

        if node.label is not None:
            dot.node(str(id(node)), label=str(node.label), shape='ellipse')
        else:
            dot.node(str(id(node)), label=str(node.attribute))
            for value, child_node in node.children.items():
                child_dot = Id3.visualize_tree(child_node, dot)
                dot.edge(str(id(node)), str(id(child_node)), label=str(value))
        
        return dot