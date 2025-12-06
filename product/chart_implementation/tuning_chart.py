import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Apply the professional Seaborn theme
sns.set_theme(style="whitegrid", context="talk") 

def generate_hyperparameter_tuning():
    
    # from the results: low at 2, peak at 10, drop at 100
    k_values = [1, 2, 5, 8, 11, 15, 17, 20, 30, 55, 80, 100]
    accuracies = [0.704, .704, 0.785, 0.793, 0.803, 0.796, 0.797, 0.797, 0.796, 0.788, 0.786, 0.784]
    
    df_tune = pd.DataFrame({'k': k_values, 'Accuracy': accuracies})

    plt.figure(figsize=(11, 7))
    sns.lineplot(x='k', y='Accuracy', data=df_tune, marker='o', linewidth=3, color='#A168FC')

    # Highlight Peak
    plt.plot(11, 0.804, marker='o', markersize=12, color='#FFA528')
    plt.text(13, 0.803, "Optimal k=11 (80.3%)", color='#FFA528', fontweight='bold')

    plt.title("Hyperparameter Tuning: Accuracy vs. k", fontweight='bold', pad=20)
    plt.xlabel("Number of Neighbors (k)", fontweight='bold')
    plt.ylabel("Accuracy", fontweight='bold')
    
    plt.text(3, 0.704, "Overfitting\n(Low k)", ha='left', color='grey')
    plt.text(105, 0.784, "Underfitting\n(High k)", ha='center', color='grey')

    plt.savefig('product/chart_implementation/tuning_curve.png', dpi=300)
    print("Saved: tuning_curve.png")

if __name__ == "__main__":
    generate_hyperparameter_tuning()