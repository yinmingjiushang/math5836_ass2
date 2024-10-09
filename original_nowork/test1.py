import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def transform_input_data(df, number, norm_flag, split_size, feature_type, type_option=None):
    #global flag_name, x_train, y_train, x_test, y_test
    # flag name
    if norm_flag == 0:
        if feature_type == 'all':
            flag_name = 'all_features + unnormalized'
        elif feature_type == 'selected':
            flag_name = 'selected + unnormalized'
        else:
            print("choose 'all' or 'selected' type")
            return 0
    if norm_flag == 1:
        if feature_type == 'all':
            flag_name = 'all_features + normalized'
        elif feature_type == 'selected':
            flag_name = 'selected + normalized'
        else:
            print("choose 'all' or 'selected' type")
            return 0

    if norm_flag == 1:
        # 将 'Rings' 列暂时移除，并对其他列进行标准化
        features = df.drop(columns=['Rings'])
        rings = df['Rings']  # 保留 'Rings' 列

        # 对特征数据进行标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 将标准化后的特征转换回 DataFrame，并保留原始列名
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

        # 将 'Rings' 列与标准化后的数据合并
        df = pd.concat([features_scaled_df, rings.reset_index(drop=True)], axis=1)

    train_data, test_data = train_test_split(df, test_size=split_size, random_state=number)

    # choose features
    if feature_type == 'all':
        x_train = train_data.drop(columns=['Rings'])
        y_train = train_data['Rings']
        x_test = test_data.drop(columns=['Rings'])
        y_test = test_data['Rings']
    elif feature_type == 'selected':
        x_train = train_data[type_option]
        y_train = train_data['Rings']
        x_test = test_data[type_option]
        y_test = test_data['Rings']

    return flag_name, x_train, y_train, x_test, y_test


def linear_regression(number, flag_name, x_train, y_train, x_test, y_test, ax):
    # create LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    # 使用测试集进行预测
    y_pred = model.predict(x_test)
    # 计算回归模型的评估指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 输出回归指标
    print(f"count = exp{number}, type = {flag_name} ")
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")

    # 绘制实际值 vs 预测值的散点图，叠加到同一个图上
    ax.scatter(y_test, y_pred, alpha=0.3, label=f'Exp {number}')
    return rmse, r2


def main():
    # 模拟数据生成，用来替代 achieve_data 逻辑
    def data_cleaning(df):
        sex_mapping = {'M': 0, 'F': 1, 'I': 2}
        df['Sex'] = df['Sex'].map(sex_mapping)
        return df

    # 模拟数据加载和处理
    csv_file_path = "../data/abalone_data.csv"
    abalone = pd.read_csv(csv_file_path)
    abalone = data_cleaning(abalone)

    # 实验设置
    num_experiments = 30
    split_size = 0.4
    threshold = 7
    norm_flag = 0
    feature_type = 'all'
    number = 0
    type_option = None

    # 创建一个图形来叠加所有实验的图1
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], color="red", linewidth=2, linestyle='--', label='y=x')  # 参考线 y=x

    for number in range(num_experiments):
        print(f"Running experiment {number + 1}/{num_experiments}")
        # q1.5: 60/40 train/test split
        train_data, test_data = train_test_split(abalone, test_size=split_size, random_state=number)

        # 数据预处理 (模拟标准化和特征选择)
        flag_name, x_train, y_train, x_test, y_test = transform_input_data(abalone, number, norm_flag, split_size,
                                                                           feature_type, type_option)

        # q2.1: linear regression
        rmse, r2 = linear_regression(number, flag_name, x_train, y_train, x_test, y_test, ax)

        print("----------------------------------------------")

    # 调整图形
    ax.set_xlabel('True Values (Rings)')
    ax.set_ylabel('Predicted Values (Rings)')
    ax.set_title('Actual vs Predicted Values (Linear Regression) - 30 Experiments')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # 显示所有叠加的图形
    plt.show()


if __name__ == "__main__":
    main()
