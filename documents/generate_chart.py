import matplotlib.pyplot as plt

def plot_class_imbalance():
    # the data
    labels = ['Non-Default (Paid)', 'Default']
    sizes = [78, 22] # 78% vs 22%
    colors = ["#A168FC", "#FFA528"] 
    
    plt.figure(figsize=(7, 7), dpi=300) # High quality for presentation
    plt.pie(sizes, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', # show percentages
        startangle=100,
        textprops={'fontsize': 14, 'weight': 'bold'})

    # add Title
    plt.title('Dataset Class Imbalance', fontsize=16, fontweight='bold')

    # save the file
    filename = 'class_imbalance_chart.png'
    plt.savefig(filename, transparent=True)
    plt.show()
if __name__ == "__main__":
    plot_class_imbalance()