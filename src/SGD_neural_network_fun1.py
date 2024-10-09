import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def tensor_SGD_regression(df, split_size, num_experiments):
    def tensor_SGD_reg(df, split_size, experiment, learning_rate, hidden_layers, neurons_per_layer):
        x = df.drop(columns=['Rings'])
        y = df['Rings']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size, random_state=experiment)

        # 标准化自变量
        scaler_x = StandardScaler()
        x_train_scaled = scaler_x.fit_transform(x_train)
        x_test_scaled = scaler_x.transform(x_test)

        # 标准化目标变量
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

        def SGD_create_model(learning_rate, hidden_layers, neurons_per_layer):
            model = Sequential()
            # 输入层 + 第一隐藏层
            model.add(Input(shape=(x_train_scaled.shape[1],)))
            model.add(Dense(neurons_per_layer, activation='relu'))

            # 添加隐藏层
            for _ in range(hidden_layers - 1):
                model.add(Dense(neurons_per_layer, activation='relu'))

            # 输出层（回归任务，线性输出）
            model.add(Dense(1, activation='linear'))

            # 使用随机梯度下降（SGD）优化器
            optimizer = SGD(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mean_squared_error')

            return model

        # 创建模型
        model = SGD_create_model(learning_rate=learning_rate, hidden_layers=hidden_layers,
                                 neurons_per_layer=neurons_per_layer)

        # 训练模型
        model.fit(x_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0)

        # 训练集预测
        y_train_pred_scaled = model.predict(x_train_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)

        # 测试集预测
        y_test_pred_scaled = model.predict(x_test_scaled)
        y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

        # 计算训练集 RMSE 和 R²
        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        # 计算测试集 RMSE 和 R²
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        return train_rmse, test_rmse, train_r2, test_r2


    # 初始化时遍历参数组合，选择最优参数
    def find_best_hyperparameters(df, split_size):
        learning_rates = [0.01, 0.001, 0.0001]
        hidden_layers_list = [1, 2, 3]
        neurons_per_layer_list = [32, 64, 128]

        best_test_rmse = float('inf')
        best_params = {}

        # 存储所有的结果
        all_results = []

        # 遍历所有参数组合
        for learning_rate in learning_rates:
            for hidden_layers in hidden_layers_list:
                for neurons_per_layer in neurons_per_layer_list:
                    print(
                        f"Testing: learning_rate={learning_rate}, hidden_layers={hidden_layers}, neurons_per_layer={neurons_per_layer}")
                    train_rmse, test_rmse, train_r2, test_r2 = tensor_SGD_reg(df, split_size, 0,learning_rate,
                                                                                     hidden_layers, neurons_per_layer)

                    # 记录当前参数组合的结果
                    result = {
                        'learning_rate': learning_rate,
                        'hidden_layers': hidden_layers,
                        'neurons_per_layer': neurons_per_layer,
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2
                    }
                    all_results.append(result)

                    # 如果当前参数组合的表现更好，则更新最佳参数
                    if test_rmse < best_test_rmse:
                        best_test_rmse = test_rmse
                        best_params = result

        # 将所有结果保存到CSV文件
        all_results_df = pd.DataFrame(all_results)
        all_results_df.to_csv("../results/fun1_param_results.csv", index=False)

        print(f"Best Parameters: {best_params}")
        return best_params


    # 进行初始化参数搜索
    best_params = find_best_hyperparameters(df, split_size=split_size)

    # 根据找到的最优参数，进行30次实验并保存到CSV文件
    train_rmse_list = []
    test_rmse_list = []
    train_r2_list = []
    test_r2_list = []
    experiment_results = []

    for experiment in range(num_experiments):
        print(f"Experiment {experiment + 1}/{num_experiments}")
        train_rmse, test_rmse, train_r2, test_r2 = tensor_SGD_reg(df, split_size, experiment,
                                                                  learning_rate=best_params['learning_rate'],
                                                                  hidden_layers=best_params['hidden_layers'],
                                                                  neurons_per_layer=best_params['neurons_per_layer'])

        # 记录每次实验的结果
        train_rmse_list.append(train_rmse)
        test_rmse_list.append(test_rmse)
        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)

        # 保存单次实验结果
        experiment_results.append({
            'experiment_num': experiment,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        })

    # 将30次实验的结果保存到CSV文件
    experiment_results_df = pd.DataFrame(experiment_results)
    experiment_results_df.to_csv("../results/fun1_experiment_results.csv", index=False)

    # 输出实验结果汇总
    avg_train_rmse = np.mean(train_rmse_list)
    avg_test_rmse = np.mean(test_rmse_list)
    avg_train_r2 = np.mean(train_r2_list)
    avg_test_r2 = np.mean(test_r2_list)

    print(f"Average train_RMSE: {avg_train_rmse}")
    print(f"Average test_RMSE: {avg_test_rmse}")
    print(f"Average train_R²: {avg_train_r2}")
    print(f"Average test_R²: {avg_test_r2}")

    return avg_train_rmse, avg_test_rmse, avg_train_r2, avg_test_r2