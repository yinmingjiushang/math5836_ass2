import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def torch_SGD_regression(split_size, num_experiments):
    csv_file_path = "../data/abalone_data.csv"
    data = pd.read_csv(csv_file_path)
    data['Sex'] = data['Sex'].map({'M': 0, 'F': 1, 'I': 2})

    X = data.drop(columns=['Rings'])
    y = data['Rings']

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)

    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    class NeuralNetwork(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(NeuralNetwork, self).__init__()
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.Linear(hidden_size, output_size))
            self.activation = nn.ReLU()

        def forward(self, x):
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
            x = self.layers[-1](x)
            return x

    def r_squared(y_true, y_pred):
        ss_res = torch.sum((y_true - y_pred) ** 2)
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2.item()

    def train_and_evaluate(X_train, y_train, X_test, y_test, hidden_size, num_layers, learning_rate, epochs, patience):
        input_size = X_train.shape[1]
        output_size = 1
        model = NeuralNetwork(input_size, hidden_size, num_layers, output_size)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                predictions_train = model(X_train)
                predictions_test = model(X_test)

                train_loss = criterion(predictions_train, y_train)
                test_loss = criterion(predictions_test, y_test)

                train_rmse = torch.sqrt(train_loss).item()
                test_rmse = torch.sqrt(test_loss).item()

                train_r2 = r_squared(y_train, predictions_train)
                test_r2 = r_squared(y_test, predictions_test)

            if test_loss < best_loss:
                best_loss = test_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

        return best_loss, train_rmse, test_rmse, train_r2, test_r2

    learning_rates = [0.01, 0.001, 0.0001]
    num_layers_list = [1, 2, 3]
    hidden_sizes = [32, 64, 128]

    epochs = 1000
    patience = 10

    best_loss = float('inf')
    best_params = {}
    best_metrics = {}
    all_results = []

    for hidden_size in hidden_sizes:
        for num_layers in num_layers_list:
            for lr in learning_rates:
                print(f'\nTraining with hidden_size={hidden_size}, num_layers={num_layers}, learning_rate={lr}')
                X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.4,
                                                                    random_state=42)  # 初始超参数选择的固定 random_state
                test_loss, train_rmse, test_rmse, train_r2, test_r2 = train_and_evaluate(
                    X_train, y_train, X_test, y_test, hidden_size, num_layers, lr, epochs, patience
                )

                result = {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'learning_rate': lr,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2
                }
                all_results.append(result)

                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = {
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'learning_rate': lr
                    }
                    best_metrics = {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2
                    }

    # 保存所有的超参数搜索结果到CSV文件
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv("../results/fun2_param_results.csv", index=False)

    print(f'\nBest Parameters: {best_params}\n, Best Loss: {best_loss:.4f}, '
          f'Best Train RMSE: {best_metrics["train_rmse"]:.4f}, Best Test RMSE: {best_metrics["test_rmse"]:.4f}, '
          f'Best Train R²: {best_metrics["train_r2"]:.4f}, Best Test R²: {best_metrics["test_r2"]:.4f}')

    # 30次实验结果
    experiment_results = []
    for experiment in range(num_experiments):
        print(f"Experiment {experiment}/{num_experiments}")
        # 在每次实验中动态设置 random_state 为 experiment + 1
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=split_size,
                                                            random_state=experiment)

        _, train_rmse, test_rmse, train_r2, test_r2 = train_and_evaluate(
            X_train, y_train, X_test, y_test, best_params['hidden_size'], best_params['num_layers'],
            best_params['learning_rate'], epochs, patience
        )

        experiment_results.append({
            'experiment_num': experiment,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        })

    # 将30次实验的结果保存到CSV文件
    experiment_results_df = pd.DataFrame(experiment_results)
    experiment_results_df.to_csv("../results/fun2_experiment_results.csv", index=False)

    print(f"Average train_RMSE: {sum([res['train_rmse'] for res in experiment_results]) / len(experiment_results)}")
    print(f"Average test_RMSE: {sum([res['test_rmse'] for res in experiment_results]) / len(experiment_results)}")
    print(f"Average train_R²: {sum([res['train_r2'] for res in experiment_results]) / len(experiment_results)}")
    print(f"Average test_R²: {sum([res['test_r2'] for res in experiment_results]) / len(experiment_results)}")

    return best_metrics['train_rmse'], best_metrics['test_rmse'], best_metrics['train_r2'], best_metrics['test_r2']



