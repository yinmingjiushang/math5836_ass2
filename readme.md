# 2024_MATH5836 assignment2
group1 work  
by anqichen, hutton, mingyin  

## run
sh ./run.sh

## project structure
```bash
├── data
│   ├── abalone_data.csv      # dataset(original)
│   └── abalone_hash.txt      # Hash verification file for ensuring data integrity
├── original_nowork                                                                                             
│   ├── anqi_code.ipynb       # original code from group member
│   ├── hutton_code.py        # original code from group member
│   ├── test1.py              # test
│   └── example.word          # example word
├── out
│   ├── linear_all_norm       # output for linear model with all features normalized
│   ├── linear_all_unnorm
│   ├── linear_sel_norm
│   ├── linear_sel_unnorm
│   ├── logistic_all_norm     # output for logistic regression with all features normalized
│   ├── logistic_all_unnorm
│   ├── logistic_sel_norm
│   ├── logistic_sel_unnorm
│   └── model_comp            # Folder containing model comparison results, including AUC, ROC curves, etc
│       ├── auc_score_comparison.png
│       ├── fun1.png
│       ├── fun2.png
│       ├── fun3.png
│       ├── linear_comp.png
│       ├── linear_vs_neural_plot.png
│       ├── roc_curve_experiment_0.png
│       └── roc_curve_experiment_x.png
├── read
├── readme.txt
├── report
│   ├── conference-template-letter.docx
│   └── report_r1p0.docx
├── results
│   ├── fun1_experiment_results.csv
│   ├── fun2_experiment_results.csv
│   ├── fun2_param_results.csv
│   ├── fun3_experiment_results.csv
│   ├── linear_analysis_results.csv
│   ├── linear_experiment_results.csv
│   ├── logistic_auc_analysis.csv
│   ├── logistic_experiment_results.csv
│   ├── neural_analysis_results.csv
│   └── neural_network_results.csv
├── src
│   ├── SGD_neural_network_fun1.py    # Code for training SGD neural network with tensorflow
│   ├── SGD_neural_network_fun2.py    # Code for training SGD neural network with pytorch
│   ├── SGD_neural_network_fun3.py    # Code for training SGD neural network with sklearn
│   ├── achieve_data.py               # Script for data processing or data retrieval
│   ├── main.py                       # contain main work
│   ├── model_comp.py                 # Model comparison script, performs comparative analysis
│   ├── requirements.txt              # List of dependencies for the Python environment
│   └── set_up.py                     # File describing the project directory structure
└── tree.txt

