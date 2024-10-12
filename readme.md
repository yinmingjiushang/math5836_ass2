# 2024_MATH5836 assignment2
group1 work  
by anqichen, hutton, mingyin  

## run
sh ./run.sh

## project structure
```bash
math5836_ass2/
├── data
│   ├── abalone_data.csv            # Dataset (original)
│   └── abalone_hash.txt            # Hash verification file for ensuring data integrity
├── original_code
│   ├── anqi_code.ipynb             # Original code from group member
│   ├── hutton_code.py              # Original code from group member
│   ├── test1.py                    # Test file
│   └── example.docx                # Example document
├── out
│   ├── linear_all_norm             # Output for linear model with all features normalized
│   ├── linear_all_unnorm           # Output for linear model with all features unnormalized
│   ├── linear_sel_norm             # Output for linear model with selected features normalized
│   ├── linear_sel_unnorm           # Output for linear model with selected features unnormalized
│   ├── logistic_all_norm           # Output for logistic regression with all features normalized
│   ├── logistic_all_unnorm         # Output for logistic regression with all features unnormalized
│   ├── logistic_sel_norm           # Output for logistic regression with selected features normalized
│   ├── logistic_sel_unnorm         # Output for logistic regression with selected features unnormalized
│   └── model_comp                  # Folder containing model comparison results, including AUC, ROC curves, etc.
│       ├── auc_score_comparison.png
│       ├── fun1.png
│       ├── fun2.png
│       ├── fun3.png
│       ├── linear_comp.png
│       ├── linear_vs_neural_plot.png
│       ├── roc_curve_experiment_0.png
│       └── roc_curve_experiment_x.png
├── readme.txt                      # Project readme file
├── report
│   ├── conference-template-letter.docx # Conference template letter
│   └── report_r1p0.docx            # Initial version of the project report
├── results
│   ├── funx_experiment_results.csv      # Record multiple experimental results of funx
│   ├── funx_param_results.csv           # Record the hyperparameter combination results of funx
│   ├── linear_analysis_results.csv      # Calculate the minimum, average, and std of the rmse and r2 results of multiple experiments
│   ├── linear_experiment_results.csv    # Record multiple experimental results of linear reg
│   ├── logistic_auc_analysis.csv        # Calculate the minimum, average, and std of logistic auc
│   ├── logistic_experiment_results.csv  # Record multiple experimental results of logistic reg
│   ├── neural_analysis_results.csv      # Calculate the minimum, average, and std of funx
│   └── neural_network_results.csv       
├── src
│   ├── SGD_neural_network_fun1.py  # Code for training SGD neural network with TensorFlow
│   ├── SGD_neural_network_fun2.py  # Code for training SGD neural network with PyTorch
│   ├── SGD_neural_network_fun3.py  # Code for training SGD neural network with scikit-learn
│   ├── achieve_data.py             # Script for data processing or data retrieval
│   ├── main.py                     # Main script for the project
│   ├── model_comp.py               # Model comparison script, performs comparative analysis
│   ├── requirements.txt            # List of dependencies for the Python environment
│   └── set_up.py                   # File describing the project directory structure
├── latex
│   ├── report.tex                  # LaTeX source file for the project paper
│   └── references.bib              # Bibliography file for references
```

## TODO
Sorry, due to time and engineering reasons, this code is still not fully modularized and macro controlled
1. Due to the execution of ed, the optimal hyperparameter selection and repeated execution of the neural network function part have not been decoupled. The current code is slightly different in ed and github
2. The selection of the normalization function is not currently controlled by macros in the code, but manually annotated. Please pay attention to the StandardScaler and MinMaxScaler related functions in the code
