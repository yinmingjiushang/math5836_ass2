import subprocess
import sys

# 要安装的依赖包及其版本
required_packages = [
    "auto_mix_prep==0.2.0",
    "matplotlib==3.9.2",
    "numpy==2.1.2",
    "pandas==2.2.3",
    "scikit_learn==1.5.2",
    "seaborn==0.13.2",
    "tensorflow==2.17.0",
    "torch==2.4.1",
    "ucimlrepo==0.0.7"
]

# 安装依赖包
def install_packages():
    for package in required_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while installing {package}: {e}")

if __name__ == "__main__":
    install_packages()
