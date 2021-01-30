import matplotlib.pyplot as plt



def plot_learning_curve(transitions, rewards, stds, label):
    # plt.figure(figsize=(12, 8))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(useMathText=True)
    
    plt.plot(transitions, rewards, label=label)
    plt.fill_between(transitions, rewards - stds, rewards + stds, alpha=0.2)
    
    plt.xlabel("Transitions")
    plt.ylabel("Mean reward")

    plt.legend()
    plt.savefig(f"plots/{label}.png")