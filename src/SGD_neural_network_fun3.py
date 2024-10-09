import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import random

def sklearn_SGD_regression(split_size, num_experiments):
    column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                    'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

    abalone_data = pd.read_csv('../data/abalone_data.csv', names=column_names, header=None)
    # 将 'Sex' 列映射为数值
    abalone_data['Sex'] = abalone_data['Sex'].map({'M': 0, 'F': 1, 'I': 2})
    # 将所有列转换为数值型，并删除任何包含 NaN 的行
    abalone_data = abalone_data.apply(pd.to_numeric, errors='coerce')
    abalone_data = abalone_data.dropna()

    def split_data(split_size, run_num):
        X = abalone_data.drop('Rings', axis=1)
        y = abalone_data['Rings']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=run_num)
        return X_train, X_test, y_train, y_test

    def scikit_nn_mod(x_train, x_test, y_train, y_test, hidden_layers=(30,), learning_rate=0.001):
        random.seed(100)
        mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layers, solver='sgd', learning_rate_init=learning_rate, max_iter=1000, random_state=10)
        mlp_model.fit(x_train, y_train)

        # 预测训练集和测试集
        y_train_pred = mlp_model.predict(x_train)
        y_test_pred = mlp_model.predict(x_test)

        # 计算训练集和测试集的 RMSE 和 R-squared
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        return train_rmse, test_rmse, train_r2, test_r2

    def select_best_parameters():
        hidden_layer_configs = [(5,), (10,), (5, 10), (10, 10), (5, 5, 5), (5, 5, 10), (5, 5), (7,), (10, 5), (10, 5, 5)]
        learning_rates = [0.01, 0.005, 0.001]

        best_train_rmse = float('inf')
        best_test_rmse = float('inf')
        best_train_r2 = -float('inf')
        best_test_r2 = -float('inf')
        best_params = None
        hyperparam_results = []

        for hidden_layers in hidden_layer_configs:
            for learning_rate in learning_rates:
                print(f"Testing: Hidden Layers={hidden_layers}, Learning Rate={learning_rate}")
                X_train, X_test, y_train, y_test = split_data(split_size, 999)  # 固定的 random_state 进行初始超参数选择

                train_rmse, test_rmse, train_r2, test_r2 = scikit_nn_mod(X_train, X_test, y_train, y_test, hidden_layers, learning_rate)

                # 记录每次超参组合的结果
                hyperparam_results.append({
                    'hidden_layers': hidden_layers,
                    'learning_rate': learning_rate,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                })

                if test_rmse < best_test_rmse:
                    best_train_rmse = train_rmse
                    best_test_rmse = test_rmse
                    best_train_r2 = train_r2
                    best_test_r2 = test_r2
                    best_params = (hidden_layers, learning_rate)

        # 将超参数搜索结果保存到CSV文件
        hyperparam_results_df = pd.DataFrame(hyperparam_results)
        hyperparam_results_df.to_csv("../results/fun3_param_results.csv", index=False)

        return best_train_rmse, best_test_rmse, best_train_r2, best_test_r2, best_params

    # 超参数搜索阶段
    best_train_rmse, best_test_rmse, best_train_r2, best_test_r2, best_params = select_best_parameters()

    print(f'\nBest Parameters: {best_params}, Best Train RMSE: {best_train_rmse:.4f}, Best Test RMSE: {best_test_rmse:.4f}')
    print(f'Best Train R²: {best_train_r2:.4f}, Best Test R²: {best_test_r2:.4f}')

    # 30次实验阶段
    experiment_results = []
    for experiment in range(num_experiments):
        print(f"\nRunning experiment {experiment + 1}/{num_experiments}")
        X_train, X_test, y_train, y_test = split_data(split_size, experiment)  # 使用实验编号作为 random_state

        train_rmse, test_rmse, train_r2, test_r2 = scikit_nn_mod(X_train, X_test, y_train, y_test, best_params[0], best_params[1])

        experiment_results.append({
            'experiment_num': experiment,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        })

    # 将30次实验结果保存到CSV文件
    experiment_results_df = pd.DataFrame(experiment_results)
    experiment_results_df.to_csv("../results/fun3_experiment_results.csv", index=False)

    # 输出30次实验的平均结果
    avg_train_rmse = sum([res['train_rmse'] for res in experiment_results]) / len(experiment_results)
    avg_test_rmse = sum([res['test_rmse'] for res in experiment_results]) / len(experiment_results)
    avg_train_r2 = sum([res['train_r2'] for res in experiment_results]) / len(experiment_results)
    avg_test_r2 = sum([res['test_r2'] for res in experiment_results]) / len(experiment_results)

    print(f"\nAverage Train RMSE: {avg_train_rmse:.4f}, Average Test RMSE: {avg_test_rmse:.4f}")
    print(f"Average Train R²: {avg_train_r2:.4f}, Average Test R²: {avg_test_r2:.4f}")

    return avg_train_rmse, avg_test_rmse, avg_train_r2, avg_test_r2


