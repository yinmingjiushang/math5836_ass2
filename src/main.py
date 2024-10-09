import os
import shutil
from xml.sax.handler import all_features

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler

import SGD_neural_network_fun1,SGD_neural_network_fun2,SGD_neural_network_fun3

# ============================
# local = 0 or ed = 1
ed_state = 0
# ============================

if ed_state == 0:
    import achieve_data

# output init
q21_linear_out_path = "../out/linear_all_unnorm"
q21_logistic_out_path = "../out/logistic_all_unnorm"
q22_linear_out_path = "../out/linear_all_norm"
q22_logistic_out_path = "../out/logistic_all_norm"
all_features_compare_out_path = "../out/logistic_all_compare"
q23_linear_unnorm_out_path = "../out/linear_sel_unnorm"
q23_logistic_unnorm_out_path = "../out/logistic_sel_unnorm"
q23_linear_norm_out_path = "../out/linear_sel_norm"
q23_logistic_norm_out_path = "../out/logistic_sel_norm"
sel_features_compare_out_path = "../out/logistic_sel_compare"

if ed_state == 0:
    if os.path.exists("../out"):
        shutil.rmtree("../out")
    if os.path.exists("../results"):
        shutil.rmtree("../results")
    achieve_data.mkdir("../results")
    achieve_data.mkdir(q21_linear_out_path)
    achieve_data.mkdir(q21_logistic_out_path)
    achieve_data.mkdir(q22_linear_out_path)
    achieve_data.mkdir(q22_logistic_out_path)
    achieve_data.mkdir(all_features_compare_out_path)
    achieve_data.mkdir(q23_linear_unnorm_out_path)
    achieve_data.mkdir(q23_logistic_unnorm_out_path)
    achieve_data.mkdir(q23_linear_norm_out_path)
    achieve_data.mkdir(q23_logistic_norm_out_path)
    achieve_data.mkdir(sel_features_compare_out_path)

# results_df
results_df = pd.DataFrame(columns=['experiment_num', 'feature_type', 'norm_flag',
                                   'train_rmse', 'test_rmse', 'train_r2', 'test_r2'])

results_logistic_df = pd.DataFrame(columns=['experiment_num', 'feature_type', 'norm_flag',
                                   'auc_score', 'fpr', 'tpr'])



# 设置显示的最大列数为 None（显示所有列）
# pd.set_option('display.max_columns', None)

def data_cleaning(df):
    # 创建字典，将字符映射为数字
    sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    # 使用map函数将sex列中的字符替换为数字
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df

def heatmap(df):
    # 计算相关性矩阵
    corr_matrix = df.corr()
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    plt.title('Heatmap', fontsize=16)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm',square=True)
    # 显示热图
    plt.show()

def corr_with_rings(df):
    # 计算相关矩阵
    corr_matrix = df.corr()
    # # 对 Rings 的相关性进行排序，以确定最正和最负的相关性
    # correlation_with_rings = corr_matrix['Rings'].drop('Rings').sort_values()
    # # 相关性最负的特征
    # neg_corr_feature = correlation_with_rings.idxmin()
    # # 相关性最正的特征
    # pos_corr_feature = correlation_with_rings.idxmax()
    # 对 Rings 的相关性取绝对值，并进行排序，以确定绝对值最大和第二大的相关性
    correlation_with_rings = corr_matrix['Rings'].drop('Rings').abs().sort_values(ascending=False)
    # 绝对值相关性最大的两个特征
    top_corr_features = correlation_with_rings.index[:2]

    #return neg_corr_feature,pos_corr_feature
    return top_corr_features[1], top_corr_features[0]

def scatter_plot_with_ring(df):

    neg_corr_feature, pos_corr_feature = corr_with_rings(df)
    # scatter_plot
    plt.figure(figsize=(14, 6))

    # sub 1：Most positively correlated features
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=df[pos_corr_feature], y=df['Rings'], alpha=0.3)
    plt.title(f'Most positively correlated features\nScatter Plot of {pos_corr_feature} vs Rings ')
    plt.xlabel(pos_corr_feature)
    plt.ylabel('Rings')

    # sub 2：Most negatively correlated features
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=df[neg_corr_feature], y=df['Rings'], alpha=0.1)
    plt.title(f'Most negatively correlated features\nScatter Plot of {neg_corr_feature} vs Rings ')
    plt.xlabel(neg_corr_feature)
    plt.ylabel('Rings')

    # 显示图像
    plt.tight_layout()
    plt.show()

def hist_plot_with_ring(df):

    neg_corr_feature, pos_corr_feature = corr_with_rings(df)
    # histplot
    plt.figure(figsize=(14, 6))

    # sub 1：Most positively correlated features
    plt.subplot(1, 3, 1)
    sns.histplot(df[pos_corr_feature], bins=30, kde=True,  alpha=0.5)
    plt.title(f'Histogram of {pos_corr_feature}')

    # sub 2：Most negatively correlated features
    plt.subplot(1, 3, 2)
    sns.histplot(df[neg_corr_feature], bins=30, kde=True,  alpha=0.5)
    plt.title(f'Histogram of {neg_corr_feature}')

    # sub 3：rings
    plt.subplot(1, 3, 3)
    sns.histplot(df['Rings'], bins=30, kde=True, alpha=0.5)
    plt.title('Histogram of Rings ')

    # 显示图形
    plt.tight_layout()
    plt.show()

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
        # features = df.drop(columns=['Rings'])
        # rings = df['Rings']  # 保留 'Rings' 列
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)

        # 使用 MinMaxScaler 对所有列进行规范化
        scaler = MinMaxScaler()
        # 将 'Rings' 列暂时移除，并对其他特征列进行规范化
        features = df.drop(columns=['Rings'])
        rings = df['Rings']  # 保留 'Rings' 列
        # 对特征数据进行 MinMaxScaler 规范化
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

def generate_output_path(base_dir, number, norm_flag, feature_type, model_type):
    model_str = "linear" if model_type == 'linear' else "logistic"
    norm_str = "norm" if norm_flag == 1 else "unnorm"
    feature_str = "all" if feature_type == 'all' else "selected"
    file_name = f"{base_dir}/{model_str}_{feature_str}_{norm_str}_exp_{number}.png"
    return file_name

def linear_regerssion(number, flag_name, x_train, y_train, x_test, y_test, file_name):
    # create LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    # 使用测试集进行预测
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    # 计算回归模型的评估指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    # 输出回归指标
    # print("---------------------")
    print(f"count = exp{number}, type = {flag_name} ")
    print(f"RMSE_train: {train_rmse}")
    print(f"R^2_train: {train_r2}")
    print(f"RMSE_test: {test_rmse}")
    print(f"R^2_test: {test_r2}")

    # 绘制实际值 vs 预测值的散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred,  alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)  # 参考线：y=x
    plt.title(f'Actual vs Predicted Values\n{flag_name}_exp{number}')
    plt.xlabel('Actual Rings')
    plt.ylabel('Predicted Rings')
    # store to local
    plt.savefig(file_name)
    plt.close()

    return train_rmse, test_rmse, train_r2, test_r2

def single_logistic_regression(number, flag_name, x_train, y_train, x_test, y_test, threshold, file_name):
    #
    y_train_bin = (y_train > threshold).astype(int)
    y_test_bin = (y_test > threshold).astype(int)
    # Logistic Regression Model
    logistic_model = LogisticRegression()
    logistic_model.fit(x_train, y_train_bin)
    y_pred_logistic = logistic_model.predict(x_test)

    # calculate AUC and ROC
    y_pred_prob = logistic_model.predict_proba(x_test)[:, 1]
    auc_score = roc_auc_score(y_test_bin, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob)

    # AUC score
    print(f'AUC Score: {auc_score}')

    # ROC plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC - AUC = {auc_score:.2f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison\n{flag_name}_exp{number}')
    plt.legend(loc="lower right")
    # store to local
    plt.savefig(file_name)
    plt.close()

    # Return AUC score and ROC data
    return auc_score, fpr, tpr

# def compare_logistic_regression(number, flag_name, x_train, y_train, x_test, y_test, threshold, file_name):
#     # 将目标变量二值化
#     y_train_bin = (y_train > threshold).astype(int)
#     y_test_bin = (y_test > threshold).astype(int)
#
#     # 模型 1：未标准化的数据
#     logistic_model_1 = LogisticRegression()
#     logistic_model_1.fit(x_train, y_train_bin)
#     y_pred_prob_1 = logistic_model_1.predict_proba(x_test)[:, 1]
#     auc_score_1 = roc_auc_score(y_test_bin, y_pred_prob_1)
#     fpr_1, tpr_1, _ = roc_curve(y_test_bin, y_pred_prob_1)
#
#     # 模型 2：标准化的数据
#     scaler = MinMaxScaler()
#     x_train_scaled = scaler.fit_transform(x_train)
#     x_test_scaled = scaler.transform(x_test)
#     logistic_model_2 = LogisticRegression()
#     logistic_model_2.fit(x_train_scaled, y_train_bin)
#     y_pred_prob_2 = logistic_model_2.predict_proba(x_test_scaled)[:, 1]
#     auc_score_2 = roc_auc_score(y_test_bin, y_pred_prob_2)
#     fpr_2, tpr_2, _ = roc_curve(y_test_bin, y_pred_prob_2)
#
#     # 打印 AUC 分数
#     print(f'AUC Score (Unnormalized): {auc_score_1}')
#     print(f'AUC Score (Normalized): {auc_score_2}')
#
#     # 绘制 ROC 曲线比较
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr_1, tpr_1, label=f'ROC (Unnormalized) - AUC = {auc_score_1:.2f}')
#     plt.plot(fpr_2, tpr_2, label=f'ROC (Normalized) - AUC = {auc_score_2:.2f}')
#     plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve Comparison\n{flag_name}_exp{number}')
#     plt.legend(loc="lower right")
#     # store to local
#     plt.savefig(file_name)
#     plt.close()

def results_analysis(results_df):
    # 使用 groupby 和 agg 来计算同一 feature_type 和 norm_flag 下的最小值、平均值和标准差
    grouped_results = results_df.groupby(['feature_type', 'norm_flag']).agg({
        'train_rmse': ['min', 'mean', 'std'],
        'test_rmse': ['min', 'mean', 'std'],
        'train_r2': ['min', 'mean', 'std'],
        'test_r2': ['min', 'mean', 'std']
    }).reset_index()

    # 重命名列名
    grouped_results.columns = ['feature_type', 'norm_flag',
                               'min_train_rmse', 'mean_train_rmse', 'std_train_rmse',
                               'min_test_rmse', 'mean_test_rmse', 'std_test_rmse',
                               'min_train_r2', 'mean_train_r2', 'std_train_r2',
                               'min_test_r2', 'mean_test_r2', 'std_test_r2']

    grouped_results.to_csv("../results/linear_analysis_results.csv", index=False)

    # 打印分组后的结果
    print(grouped_results)


def main():
    # 执行加载或更新data
    achieve_data.achieve_data_main()

    # 从csv读取数据
    csv_file_path = "../data/abalone_data.csv"
    abalone = pd.read_csv(csv_file_path)

    # q1.1: data cleaning
    abalone = data_cleaning(abalone)

    # q1.2: heat map
    heatmap(abalone)

    # q1.3: scatter_plot_with_ring
    scatter_plot_with_ring(abalone)

    # q1.4: hist_plot_with_ring
    hist_plot_with_ring(abalone)

    # init
    number = 0
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    flag_name = None
    feature_type = 'all'
    type_option = None
    file_name = None
    num_experiments = 30
    split_size = 0.4
    threshold = 7
    norm_flag = 0



    # q1.5: 60/40 train/test split
    train_data, test_data = train_test_split(abalone, test_size=split_size, random_state=number)

    for number in range(num_experiments):
        print("===============================================")
        # q2.1: linear regression
        norm_flag = 0
        feature_type = 'all'
        flag_name, x_train, y_train, x_test, y_test = transform_input_data(abalone, number, norm_flag, split_size, feature_type, type_option)
        # linear regression
        file_name = generate_output_path(q21_linear_out_path, number, norm_flag, feature_type, "linear")
        train_rmse, test_rmse, train_r2, test_r2 = linear_regerssion(number, flag_name, x_train, y_train, x_test, y_test, file_name)
        results_df.loc[len(results_df)] = [number, feature_type, norm_flag, train_rmse, test_rmse, train_r2, test_r2]
        # logistic regression
        file_name = generate_output_path(q21_logistic_out_path,number, norm_flag, feature_type, "logistic")
        auc_score, fpr, tpr = single_logistic_regression(number, flag_name, x_train, y_train, x_test, y_test, threshold, file_name)
        results_logistic_df.loc[len(results_logistic_df)] = [number, feature_type, norm_flag,auc_score, fpr, tpr]
        print("----------------------------------------------")

        # q2.2: normalising input data
        norm_flag = 1
        feature_type = 'all'
        flag_name, x_train, y_train, x_test, y_test = transform_input_data(abalone, number, norm_flag, split_size, feature_type, type_option)
        # linear regression
        file_name = generate_output_path(q22_linear_out_path, number, norm_flag, feature_type, "linear")
        train_rmse, test_rmse, train_r2, test_r2 = linear_regerssion(number, flag_name, x_train, y_train, x_test, y_test, file_name)
        results_df.loc[len(results_df)] = [number, feature_type, norm_flag, train_rmse, test_rmse, train_r2, test_r2]
        # logistic regression
        file_name = generate_output_path(q22_logistic_out_path, number, norm_flag, feature_type, "logistic")
        auc_score, fpr, tpr = single_logistic_regression(number, flag_name, x_train, y_train, x_test, y_test, threshold, file_name)
        results_logistic_df.loc[len(results_logistic_df)] = [number, feature_type, norm_flag, auc_score, fpr, tpr]
        print("----------------------------------------------")

        # q2.3: two selected input features
        # unnorm
        norm_flag = 0
        neg_corr_feature, pos_corr_feature = corr_with_rings(abalone)
        feature_type = 'selected'
        type_option = [neg_corr_feature, pos_corr_feature]
        flag_name, x_train, y_train, x_test, y_test = transform_input_data(abalone, number, norm_flag, split_size, feature_type, type_option)
        # linear regression
        file_name = generate_output_path(q23_linear_unnorm_out_path, number, norm_flag, feature_type, "linear")
        train_rmse, test_rmse, train_r2, test_r2 = linear_regerssion(number, flag_name, x_train, y_train, x_test, y_test, file_name)
        results_df.loc[len(results_df)] = [number, feature_type, norm_flag, train_rmse, test_rmse, train_r2, test_r2]
        # logistic regression
        file_name = generate_output_path(q23_logistic_unnorm_out_path, number, norm_flag, feature_type, "logistic")
        auc_score, fpr, tpr = single_logistic_regression(number, flag_name, x_train, y_train, x_test, y_test, threshold, file_name)
        results_logistic_df.loc[len(results_logistic_df)] = [number, feature_type, norm_flag, auc_score, fpr, tpr]

        # norm
        norm_flag = 1
        # linear regression
        file_name = generate_output_path(q23_linear_norm_out_path, number, norm_flag, feature_type, "linear")
        train_rmse, test_rmse, train_r2, test_r2 = linear_regerssion(number, flag_name, x_train, y_train, x_test, y_test, file_name)
        results_df.loc[len(results_df)] = [number, feature_type, norm_flag, train_rmse, test_rmse, train_r2, test_r2]
        # linear regression
        file_name = generate_output_path(q23_logistic_norm_out_path, number, norm_flag, feature_type, "logistic")
        auc_score, fpr, tpr = single_logistic_regression(number, flag_name, x_train, y_train, x_test, y_test, threshold, file_name)
        results_logistic_df.loc[len(results_logistic_df)] = [number, feature_type, norm_flag, auc_score, fpr, tpr]

        print("===============================================")

        plt.close()

    # 保存结果
    results_df.to_csv("../results/linear_experiment_results.csv", index=False)
    results_logistic_df.to_csv("../results/logistic_experiment_results.csv", index=False)

    # q2.4:
    # create df
    neural_network_df = pd.DataFrame(columns=['fun_name', 'train_rmse', 'test_rmse', 'train_r2', 'test_r2'])

    fun1_avg_train_rmse, fun1_avg_test_rmse, fun1_avg_train_r2, fun1_avg_test_r2 = SGD_neural_network_fun1.tensor_SGD_regression(abalone, split_size, num_experiments)
    neural_network_df.loc[len(neural_network_df)] = ["fun1", fun1_avg_train_rmse, fun1_avg_test_rmse, fun1_avg_train_r2, fun1_avg_test_r2]

    fun2_avg_train_rmse, fun2_avg_test_rmse, fun2_avg_train_r2, fun2_avg_test_r2 = SGD_neural_network_fun2.torch_SGD_regression(split_size, num_experiments)
    neural_network_df.loc[len(neural_network_df)] = ["fun2", fun2_avg_train_rmse, fun2_avg_test_rmse, fun2_avg_train_r2, fun2_avg_test_r2]

    fun3_avg_train_rmse, fun3_avg_test_rmse, fun3_avg_train_r2, fun3_avg_test_r2 = SGD_neural_network_fun3.sklearn_SGD_regression(split_size, num_experiments)
    neural_network_df.loc[len(neural_network_df)] = ["fun3", fun3_avg_train_rmse, fun3_avg_test_rmse, fun3_avg_train_r2, fun3_avg_test_r2]

    neural_network_df.to_csv("../results/neural_network_results.csv", index=False)
    # print(abalone)

    # q2.5
    results_analysis(results_df)

if __name__ == "__main__":
    main()