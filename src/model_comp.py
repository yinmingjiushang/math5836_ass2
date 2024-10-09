# Re-importing necessary libraries and reloading the files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import seaborn as sns

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  The folder exists!  ---")

file = "../out/model_comp"
mkdir(file)



def linear_vs_neural():
    # Load the CSV files for linear model results and neural network results
    linear_analysis_results_path = '../results/linear_analysis_results.csv'
    linear_experiment_results_path = '../results/linear_experiment_results.csv'
    neural_network_results_path = '../results/neural_network_results.csv'

    # Read the CSV files
    linear_analysis_results = pd.read_csv(linear_analysis_results_path)
    linear_experiment_results = pd.read_csv(linear_experiment_results_path)
    neural_network_results = pd.read_csv(neural_network_results_path)

    # Display the first few rows of both datasets to understand their structure
    linear_analysis_results.head(), linear_experiment_results.head(), neural_network_results.head()

    # Data preparation for visualization: comparing RMSE and R² between linear models and neural networks

    # Calculate the mean RMSE and R² for the linear models (aggregated from both files)
    linear_model_avg_rmse_r2 = linear_analysis_results[['mean_train_rmse', 'mean_test_rmse', 'mean_train_r2', 'mean_test_r2']].mean().to_dict()

    # Neural network mean RMSE and R²
    neural_network_avg_rmse_r2 = neural_network_results[['train_rmse', 'test_rmse', 'train_r2', 'test_r2']].mean().to_dict()

    # Data for plotting
    models = ['Linear Model', 'Neural Network']

    # RMSE comparison
    train_rmse = [linear_model_avg_rmse_r2['mean_train_rmse'], neural_network_avg_rmse_r2['train_rmse']]
    test_rmse = [linear_model_avg_rmse_r2['mean_test_rmse'], neural_network_avg_rmse_r2['test_rmse']]

    # R² comparison
    train_r2 = [linear_model_avg_rmse_r2['mean_train_r2'], neural_network_avg_rmse_r2['train_r2']]
    test_r2 = [linear_model_avg_rmse_r2['mean_test_r2'], neural_network_avg_rmse_r2['test_r2']]

    # Plotting grouped bar charts for better visual impact
    x = np.arange(len(models))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot RMSE comparison
    rects1 = ax[0].bar(x - width/2, train_rmse, width, label='Train RMSE', color='skyblue')
    rects2 = ax[0].bar(x + width/2, test_rmse, width, label='Test RMSE', color='orange')
    ax[0].set_ylabel('RMSE')
    ax[0].set_title('RMSE Comparison (Train vs Test)')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(models)
    ax[0].legend()

    # Plot R² comparison
    rects3 = ax[1].bar(x - width/2, train_r2, width, label='Train R²', color='lightgreen')
    rects4 = ax[1].bar(x + width/2, test_r2, width, label='Test R²', color='purple')
    ax[1].set_ylabel('R²')
    ax[1].set_title('R² Comparison (Train vs Test)')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(models)
    ax[1].legend()

    # Adding labels on the bars
    def add_labels(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1, ax[0])
    add_labels(rects2, ax[0])
    add_labels(rects3, ax[1])
    add_labels(rects4, ax[1])

    plt.tight_layout()
    plt.show()



def linear_comp(experiments):
    csv_file_path = "../results/linear_experiment_results.csv"
    data = pd.read_csv(csv_file_path)
    # Create a directory to save plots
    output_dir = '../out/linear_experiment_plots'
    os.makedirs(output_dir, exist_ok=True)

    def linear_experiment_comparisons_to_files():
        metrics = ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']
        metric_titles = {
            'train_rmse': 'Train RMSE',
            'test_rmse': 'Test RMSE',
            'train_r2': 'Train R2 Score',
            'test_r2': 'Test R2 Score'
        }

        # Comparing the same metric across different experiments
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            experiments = data['experiment_num'].unique()
            feature_types = data['feature_type'].unique()
            norm_flags = data['norm_flag'].unique()
            bar_width = 0.2
            index = np.arange(len(experiments))

            for i, (feature, norm) in enumerate([(ft, nf) for ft in feature_types for nf in norm_flags]):
                subset = data[(data['feature_type'] == feature) & (data['norm_flag'] == norm)]
                positions = index + (i * bar_width)
                plt.bar(
                    positions,
                    subset[metric],
                    bar_width,
                    alpha=0.6,
                    label=f'Feature: {feature}, Norm: {norm}'
                )

            plt.title(f'Comparison of {metric_titles[metric]} Across Different Experiments')
            plt.xlabel('Experiment Number')
            plt.ylabel(metric_titles[metric])
            plt.xticks(index + bar_width * (len(feature_types) * len(norm_flags) / 2), experiments, rotation=0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{metric}_comparison_across_experiments.png')
            plt.close()

        # Comparing different metrics within the same experiment
        for experiment in range(len(experiments)):
            plt.figure(figsize=(12, 6))
            bar_width = 0.2
            index = np.arange(len(metrics))

            feature_types = data['feature_type'].unique()
            norm_flags = data['norm_flag'].unique()

            for i, (feature, norm) in enumerate([(ft, nf) for ft in feature_types for nf in norm_flags]):
                subset = data[(data['experiment_num'] == experiment) & (data['feature_type'] == feature) & (data['norm_flag'] == norm)]
                positions = index + (i * bar_width)
                plt.bar(
                    positions,
                    [subset[metric].values[0] if not subset.empty else 0 for metric in metrics],
                    bar_width,
                    alpha=0.6,
                    label=f'Feature: {feature}, Norm: {norm}'
                )

            plt.title(f'Comparison of Different Metrics for Experiment {experiment}')
            plt.xlabel('Metric')
            plt.ylabel('Value')
            plt.xticks(index + bar_width * (len(feature_types) * len(norm_flags) / 2), [metric_titles[metric] for metric in metrics], rotation=0)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/metrics_comparison_experiment_{experiment}.png')
            plt.close()

    # Calling the function to save the comparison plots to files
    linear_experiment_comparisons_to_files()


def linear_comp2(csv_path,name):
    new_data = pd.read_csv(csv_path)
    # Set up the plot size and style for the new data
    plt.figure(figsize=(12, 8))

    # Plot train and test RMSE for comparison for the new data
    plt.subplot(2, 1, 1)
    for feature_type in new_data['feature_type'].unique():
        subset = new_data[new_data['feature_type'] == feature_type]
        plt.plot(subset['experiment_num'], subset['train_rmse'], label=f'{feature_type} - Train RMSE', marker='o')
        plt.plot(subset['experiment_num'], subset['test_rmse'], label=f'{feature_type} - Test RMSE', marker='o',
                 linestyle='--')
    plt.xlabel('Experiment Number')
    plt.ylabel('RMSE')
    plt.title('Train vs Test RMSE (Grouped by Feature Type)')
    plt.legend()
    plt.grid(True)

    # Plot train and test R² for comparison for the new data
    plt.subplot(2, 1, 2)
    for feature_type in new_data['feature_type'].unique():
        subset = new_data[new_data['feature_type'] == feature_type]
        plt.plot(subset['experiment_num'], subset['train_r2'], label=f'{feature_type} - Train R²', marker='o')
        plt.plot(subset['experiment_num'], subset['test_r2'], label=f'{feature_type} - Test R²', marker='o',
                 linestyle='--')
    plt.xlabel('Experiment Number')
    plt.ylabel('R²')
    plt.title('Train vs Test R² (Grouped by Feature Type)')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"../out/model_comp/{name}.png")
    plt.close()


def neural_comp(csv_path, name):
    data = pd.read_csv(csv_path)

    # Set up the plot size and style
    plt.figure(figsize=(12, 8))

    # Plot train and test RMSE for comparison
    plt.subplot(2, 1, 1)
    plt.plot(data['experiment_num'], data['train_rmse'], label='Train RMSE', marker='o')
    plt.plot(data['experiment_num'], data['test_rmse'], label='Test RMSE', marker='o', linestyle='--')
    plt.xlabel('Experiment Number')
    plt.ylabel('RMSE')
    plt.title('Train vs Test RMSE')
    plt.legend()
    plt.grid(True)

    # Plot train and test R² for comparison
    plt.subplot(2, 1, 2)
    plt.plot(data['experiment_num'], data['train_r2'], label='Train R²', marker='o')
    plt.plot(data['experiment_num'], data['test_r2'], label='Test R²', marker='o', linestyle='--')
    plt.xlabel('Experiment Number')
    plt.ylabel('R²')
    plt.title('Train vs Test R²')
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"../out/model_comp/{name}.png")
    plt.close()


linear_vs_neural()
# linear_comp(30)
linear_comp2("../results/linear_experiment_results.csv", "linear_comp")
neural_comp("../results/fun1_experiment_results.csv", "fun1")
neural_comp("../results/fun2_experiment_results.csv", "fun2")
neural_comp("../results/fun3_experiment_results.csv", "fun3")