# eda_automation_pipeline/config.py
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

def configure_environment():
    warnings.filterwarnings('ignore')
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.figsize'] = (10, 5)
