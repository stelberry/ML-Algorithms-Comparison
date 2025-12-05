import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="whitegrid", context="talk")

def plot_final_comparison():
    # define the Data
    data = {
        'Model': ['1-NN', 'Optimized k-NN (k=11)', 'Decision Tree (CART)'],
        'Accuracy': [0.73, 0.808, 0.82],
        'Color': ['#FFA528', '#A168FC', '#9BBB59']
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Accuracy", data=df, hue="Model", palette=data['Color'], legend=False)
    ax.set_ylim(0.70, 0.85) 
    
    ax.set_title("Final Model Comparison: Accuracy on Test Set", fontweight='bold', pad=20)
    ax.set_ylabel("Accuracy Score", fontweight='bold')
    ax.set_xlabel("") # No x-label needed as categories are clear

    for i, v in enumerate(df['Accuracy']):
        ax.text(i, v + 0.002, f"{v:.1%}", ha='center', fontweight='bold', fontsize=14)

    plt.tight_layout()
    plt.savefig('final_results_chart.png', dpi=300)
    print("Chart saved as final_results_chart.png")

if __name__ == "__main__":
    plot_final_comparison()