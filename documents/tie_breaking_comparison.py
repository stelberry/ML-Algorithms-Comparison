import matplotlib.pyplot as plt

def tie_breaking_comparison():
    # data 
    strategies = ['Random Selection \n(best k=11)', 'Alphabetical/Numerical Priority \n(best k=11)', 'Distance-Based \n(best k=11)']
    accuracies = [0.7229, 0.8077, 0.8077] 
    
    # setup the colors 
    colors = ['#FFA528', '#A168FC', '#9BBB59'] 

    # create the horizontal Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    bars = ax.barh(strategies, accuracies, color=colors, height=0.6)

    ax.set_xlim(0.6, 0.9)
    ax.set_xlabel('Accuracy Score', fontsize=11, fontweight='bold')
    ax.set_title('Impact of Tie-Breaking Strategy', fontsize=13, fontweight='bold')
    
    # remove unpleasent-looking borders
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # add the numbers on the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.1%}', 
                va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('tie_breaking_chart.png')
    print("Chart saved as 'tie_breaking_chart.png'")

if __name__ == "__main__":
    tie_breaking_comparison()