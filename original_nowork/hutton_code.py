import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPRegressor
import random

column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
                'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

# Load the data into a DataFrame
abalone_data = pd.read_csv('../data/abalone_data.csv', names=column_names, header=None)
abalone_data.head()

# 1.1
# Map 'Sex' column to numerical
abalone_data['Sex'] = abalone_data['Sex'].map({'M': 0, 'F': 1, 'I': 2})

# 1.2
# correlation matrix
correlation_matrix = abalone_data.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Abalone Dataset")
plt.show()


# 1.3
# scatter plot with distinct colors for each Sex
plt.figure(figsize=(8, 6))
for sex, color in zip([0, 1, 2], ['blue', 'red', 'orange']):
    label = {0: 'Male', 1: 'Female', 2: 'Infant'}[sex]
    plt.scatter(abalone_data[abalone_data['Sex'] == sex]['Shell_weight'],
                abalone_data[abalone_data['Sex'] == sex]['Rings'],
                color=color, label=label, alpha=0.6)

# title and labels
plt.title("Scatter Plot of Shell_weight vs Rings with Sex")
plt.xlabel("Shell_weight")
plt.ylabel("Rings")
plt.legend(title="Sex", loc="upper right")
plt.show()

# 1.4
# Create histograms of Whole_weight, Diameter and Rings
abalone_data[['Shell_weight', 'Diameter', 'Rings']].hist(bins=20, figsize=(12, 6))
plt.suptitle('Histograms of Shell_weight, Diameter and Rings')
plt.show()

# 1.5
# Split the dataset into training and testing ((60 : 40)
def split_data(run_num):
    X = abalone_data.drop('Rings', axis=1)
    y = abalone_data['Rings']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state= run_num)

    return X_train, X_test, y_train, y_test



#2.1
def LR(run_num):
    X_train, X_test, y_train, y_test = split_data(run_num)

    linear_reg = LinearRegression()

    linear_reg.fit(X_train, y_train)  # Train

    y_pred = linear_reg.predict(X_test)  # predictions

    # RMSE and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # plot line the pre_y v.s. test_y
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # y=x的对角线
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.show()

    # Plot residuals
    residuals = y_pred - y_test
    plt.plot(residuals, linewidth=1)

    # Output the model's coefficients, RMSE, and R-squared score
    model_coefficients = linear_reg.coef_
    model_intercept = linear_reg.intercept_

    model_results = {
        "RMSE": rmse,
        "R-squared": r2,
        "Coefficients": model_coefficients,
        "Intercept": model_intercept
    }
    return model_results


# 2.2
def LR_normalization(run_num):
    X_train, X_test, y_train, y_test = split_data(run_num)

    normalizer = Normalizer().fit(X_train)
    X_train_normalized = normalizer.transform(X_train)
    X_test_normalized = normalizer.transform(X_test)

    linear_reg = LinearRegression()
    linear_reg.fit(X_train_normalized, y_train)  # Train

    y_pred = linear_reg.predict(X_test_normalized)  # predictions

    # RMSE and R-squared
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    model_coefficients = linear_reg.coef_
    model_intercept = linear_reg.intercept_

    return {"RMSE": rmse, "R-squared": r2, "Coefficients": model_coefficients, "Intercept": model_intercept}

X_train, X_test, y_train, y_test = split_data(1)
print("Before normalization:")
print(X_train.describe())

normalizer = Normalizer().fit(X_train)
X_train_normalized = normalizer.transform(X_train)

X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns)

print("After normalization:")
print(X_train_normalized_df.describe())

# 2.3
# Select Shell_weight and Diameter
def select_LR(num):
    X_simplified = abalone_data[['Shell_weight', 'Diameter']]
    y = abalone_data['Rings']
    # Split
    X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simplified, y, test_size=0.40, random_state=num)

# Train a linear regression model on the simplified features
    linear_reg_simple = LinearRegression()
    linear_reg_simple.fit(X_train_simple, y_train_simple)

    y_pred_simple = linear_reg_simple.predict(X_test_simple)

    # Calculate RMSE and R-squared score for the simplified model
    rmse_simple = mean_squared_error(y_test_simple, y_pred_simple, squared=False)
    r2_simple = r2_score(y_test_simple, y_pred_simple)

    model_results_simple = {
        "RMSE_simplified": rmse_simple,
        "R-squared_simplified": r2_simple,
        "Coefficients_simplified": linear_reg_simple.coef_,
        "Intercept_simplified": linear_reg_simple.intercept_
    }

    return model_results_simple

print(select_LR(2))


# 2.4
def scikit_nn_mod(x_train, x_test, y_train, y_test, hidden_layers=(30,), learning_rate=0.001):
    mlp_model = MLPRegressor(hidden_layer_sizes=hidden_layers, solver='sgd', learning_rate_init=learning_rate, max_iter=1000)
    mlp_model.fit(x_train, y_train)
    y_pred = mlp_model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rsquared = r2_score(y_test, y_pred)
    return rmse, rsquared

def select_experiments():

    hidden_layer_configs = [(5,), (6,), (7,), (8,), (9,), (10)]
    learning_rates = [0.01, 0.005, 0.001]

    results = []

    # 30 times
    for experiment_num in range(30):
        print(f"Running experiment {experiment_num + 1}/30")

        hidden_layers = random.choice(hidden_layer_configs)
        learning_rate = random.choice(learning_rates)

        X_train, X_test, y_train, y_test = split_data(experiment_num)

        rmse, r2 = scikit_nn_mod(X_train, X_test, y_train, y_test, hidden_layers, learning_rate)

        result = {
            "experiment_num": experiment_num + 1,
            "hidden_layers": hidden_layers,
            "learning_rate": learning_rate,
            "RMSE": rmse,
            "R-squared": r2
        }
        results.append(result)

        print(f"Experiment {experiment_num + 1}: Hidden Layers={hidden_layers}, Learning Rate={learning_rate}, RMSE={rmse:.4f}, R-squared={r2:.4f}")

    best_result = min(results, key=lambda x: x["RMSE"])
    print("\nBest Result:")
    print(f"Best Configuration: Hidden Layers={best_result['hidden_layers']}, Learning Rate={best_result['learning_rate']}")
    print(f"Best RMSE: {best_result['RMSE']:.4f}, Best R-squared: {best_result['R-squared']:.4f}")

    return results, best_result

print(select_experiments())

# 2.5
X_train, X_test, y_train, y_test = split_data(100)
best_nn_result = scikit_nn_mod(X_train, X_test, y_train, y_test, (5,), 0.005)
result_lr = LR(100)
result_lr_norm = LR_normalization(100)
result_select_lr = select_LR(100)

print("Linear Regression:", result_lr)
print("Linear Regression with Normalization:", result_lr_norm)
print("Simplified Linear Regression:", result_select_lr)
print("Best Neural Network Result:", best_nn_result)
