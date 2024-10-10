# Re-importing necessary libraries and reloading the files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import seaborn as sns
import ast
import re

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

def find_best_param():
    # Load the CSV file to analyze
    file_path = '../results/fun1_param_results.csv'
    data = pd.read_csv(file_path)
    # Find the row with the minimum value of "test_rmse"
    min_rmse_row = data.loc[data['test_rmse'].idxmin()]
    print(min_rmse_row)

    file_path = '../results/fun2_param_results.csv'
    data = pd.read_csv(file_path)
    # Find the row with the minimum value of "test_rmse"
    min_rmse_row = data.loc[data['test_rmse'].idxmin()]
    print(min_rmse_row)

    file_path = '../results/fun3_param_results.csv'
    data = pd.read_csv(file_path)
    # Find the row with the minimum value of "test_rmse"
    min_rmse_row = data.loc[data['test_rmse'].idxmin()]
    print(min_rmse_row)


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

def logistic_auc_comp(csv_path, output_dir):
    # Load the dataset
    experiment_results = pd.read_csv(csv_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plotting AUC comparison for all experiments
    plt.figure(figsize=(12, 8))

    for feature_type in experiment_results['feature_type'].unique():
        feature_data = experiment_results[experiment_results['feature_type'] == feature_type]
        for norm_flag in feature_data['norm_flag'].unique():
            norm_data = feature_data[feature_data['norm_flag'] == norm_flag]
            plt.plot(norm_data['experiment_num'], norm_data['auc_score'], marker='o', linestyle='-', alpha=0.7,
                     label=f'Feature: {feature_type}, Norm: {norm_flag}')

    # Plot settings
    plt.xlabel('Experiment Number')
    plt.ylabel('AUC Score')
    plt.title('AUC Score Comparison for All Experiments')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.grid(True)

    # Save the plot as PNG
    plot_path = os.path.join(output_dir, 'auc_score_comparison.png')
    plt.savefig(plot_path)
    plt.close()

    # Print statement to confirm saving
    print(f"AUC score comparison plot saved to {plot_path}")

def logistic_roc_comp(csv_path, output_dir):
    # Load the dataset
    experiment_results = pd.read_csv(csv_path)

    # Function to clean and parse the list strings
    def clean_and_parse_list(list_str):
        # Removing extra whitespace and replacing it with commas where appropriate
        cleaned_str = re.sub(r'\s+', ',', list_str.strip())
        # Removing any consecutive commas introduced
        cleaned_str = re.sub(r',+', ',', cleaned_str)
        # Ensuring the string starts and ends correctly as a list
        cleaned_str = cleaned_str.replace('[,', '[').replace(',]', ']')
        # Convert to list using ast.literal_eval
        return ast.literal_eval(cleaned_str)

    # Iterate through each experiment number and plot the ROC curves
    unique_experiments = experiment_results['experiment_num'].unique()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for experiment in unique_experiments:
        experiment_data = experiment_results.loc[experiment_results['experiment_num'] == experiment].copy()

        # Applying the new parsing function to fix the columns
        experiment_data['fpr'] = experiment_data['fpr'].apply(clean_and_parse_list)
        experiment_data['tpr'] = experiment_data['tpr'].apply(clean_and_parse_list)

        # Plotting the ROC curve for each experiment
        plt.figure(figsize=(10, 8))

        for _, row in experiment_data.iterrows():
            fpr = row['fpr']
            tpr = row['tpr']
            auc_score = row['auc_score']
            label = f"Feature: {row['feature_type']}, Norm: {row['norm_flag']} - AUC = {auc_score:.2f}"
            plt.plot(fpr, tpr, label=label, linestyle='-', alpha=0.7)

        # Plot settings
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for Experiment {experiment}')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.grid(True)

        # Save the plot as PNG
        plt.savefig(os.path.join(output_dir, f'roc_curve_experiment_{experiment}.png'))
        plt.close()
        print(
            f"ROC curve for experiment {experiment} saved to {os.path.join(output_dir, f'roc_curve_experiment_{experiment}.png')}")


def logistic_auc_analysis(input_csv_path, output_folder_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_csv_path)

    # Extract AUC metrics grouped by features and normalization techniques
    group_by_cols = ['feature_type', 'norm_flag']
    auc_stats = df.groupby(group_by_cols)['auc_score'].agg(['min', 'mean', 'std']).reset_index()

    # Create output folder if it doesn't exist
    # os.makedirs(output_folder_path, exist_ok=True)

    # Save the summary DataFrame to a CSV file
    output_csv_path = os.path.join(output_folder_path, 'logistic_auc_analysis.csv')
    auc_stats.to_csv(output_csv_path, index=False)

    # Print confirmation
    print(f'Successfully saved AUC summary to {output_csv_path}')


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

def linear_vs_neural_comp():
    # Load the CSV files
    linear_results_path = '../results/linear_analysis_results.csv'
    neural_results_path = '../results/neural_network_results.csv'

    linear_df = pd.read_csv(linear_results_path)
    neural_df = pd.read_csv(neural_results_path)

    # Extracting mean values from linear regression dataset for comparison
    linear_mean_train_rmse = linear_df['mean_train_rmse'].mean()
    linear_mean_test_rmse = linear_df['mean_test_rmse'].mean()
    linear_mean_train_r2 = linear_df['mean_train_r2'].mean()
    linear_mean_test_r2 = linear_df['mean_test_r2'].mean()

    # Using the original values of each experiment for neural network functions
    neural_fun1_values = neural_df[neural_df['fun_name'] == 'fun1'][
        ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']].mean()
    neural_fun2_values = neural_df[neural_df['fun_name'] == 'fun2'][
        ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']].mean()
    neural_fun3_values = neural_df[neural_df['fun_name'] == 'fun3'][
        ['train_rmse', 'test_rmse', 'train_r2', 'test_r2']].mean()

    # Metrics for comparison
    metrics = ['Train RMSE', 'Test RMSE', 'Train R^2', 'Test R^2']
    linear_values = [linear_mean_train_rmse, linear_mean_test_rmse, linear_mean_train_r2, linear_mean_test_r2]

    # Plotting comparison of Linear Regression and each Neural Network function without specifying colors
    plt.figure(figsize=(12, 8))

    width = 0.2
    x = range(len(metrics))

    # Plotting all functions for comparison using default colors
    plt.bar([i - 1.5 * width for i in x], linear_values, width=width, label='Linear Regression')
    plt.bar([i - 0.5 * width for i in x], neural_fun1_values, width=width, label='Neural Network - Fun1')
    plt.bar([i + 0.5 * width for i in x], neural_fun2_values, width=width, label='Neural Network - Fun2')
    plt.bar([i + 1.5 * width for i in x], neural_fun3_values, width=width, label='Neural Network - Fun3')

    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Comparison of Linear Regression vs Neural Network Models (Fun1, Fun2, Fun3)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(axis='y', linestyle='--', linewidth=0.7)

    # Save the plot as a file
    plt.savefig('../out/model_comp/linear_vs_neural_plot.png')
    plt.close()

def neural_results_analysis():
    # Load the CSV file
    fun1_data = pd.read_csv('../results/fun1_experiment_results.csv')
    fun2_data = pd.read_csv('../results/fun2_experiment_results.csv')
    fun3_data = pd.read_csv('../results/fun3_experiment_results.csv')

    # Calculate minimum, mean, and standard deviation for each dataset
    fun1_summary = fun1_data.describe().loc[['min', 'mean', 'std']].drop(columns=['experiment_num'])
    fun2_summary = fun2_data.describe().loc[['min', 'mean', 'std']].drop(columns=['experiment_num'])
    fun3_summary = fun3_data.describe().loc[['min', 'mean', 'std']].drop(columns=['experiment_num'])

    # Concatenate the summary statistics of all three datasets
    summary_all = pd.concat([fun1_summary, fun2_summary, fun3_summary], keys=['fun1', 'fun2', 'fun3'])

    # Write the concatenated summary statistics to a CSV file
    output_path = '../results/neural_analysis_results.csv'
    summary_all.to_csv(output_path)

    # Informing user of the file location
    output_path



#linear_vs_neural()
#find_best_param()
linear_comp2("../results/linear_experiment_results.csv", "linear_comp")
neural_comp("../results/fun1_experiment_results.csv", "fun1")
neural_comp("../results/fun2_experiment_results.csv", "fun2")
neural_comp("../results/fun3_experiment_results.csv", "fun3")
logistic_auc_comp("../results/logistic_experiment_results.csv", "../out/model_comp")
logistic_roc_comp("../results/logistic_experiment_results.csv", "../out/model_comp/")
logistic_auc_analysis("../results/logistic_experiment_results.csv","../results")
linear_vs_neural_comp()
neural_results_analysis()