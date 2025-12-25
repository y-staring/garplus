import matplotlib.pyplot as plt

def plot_one_curve(data_list, label, title):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(data_list) + 1), data_list, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_two_curve(list1, list2, name1, name2, label, title):
    plt.figure()
    plt.plot(range(1, len(list1) + 1), list1, label=name1)
    plt.plot(range(1, len(list2) + 1), list2, label=name2)
    plt.xlabel("Epoch")
    plt.ylabel(label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig("val_prc_two_runs.png", dpi=200)
    plt.show()