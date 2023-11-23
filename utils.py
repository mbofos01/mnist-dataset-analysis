import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects, ax):
    """
    Description
    ----------
    Label the top of the bar graph

    Parameters
    ----------
    rects
    ax

    Returns
    -------
    None
    """
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def get_statistics(n, y_true, predicted_classes, correct, incorrect):
    """
    Description
    ----------
    Counts how many predictions are correct and incorrect

    Parameters
    ----------
    incorrect
    correct
    predicted_classes
    n
    y_true

    Returns
    -------
    tuple[Any, Any]
    """
    for i in range(0, 37000):
        if y_true[i] == n and predicted_classes[i] == n:
            correct['Digit-' + str(n)] += 1
        elif y_true[i] == n and predicted_classes[i] != n:
            incorrect['Digit-' + str(n)] += 1
    return correct, incorrect


def plot_classification_report(y_true, predicted_classes, model_name):
    """
    Description
    ----------
    Builds a classification report that includes the correct/incorrect classifications per digit and exports it

    Parameters
    ----------
    model_name
    predicted_classes
    y_true

    Returns
    -------
    None
    """

    # Plot labels
    labels = ["Digit-{}".format(i) for i in range(10)]
    # Dictionaries with keys the classes
    correct = {}
    incorrect = {}

    for digit in labels:
        correct[digit] = 0
        incorrect[digit] = 0
    for c in range(0, 10):
        get_statistics(c, y_true, predicted_classes, correct, incorrect)
        corrects = []  # list of correct predictions per classes
    for v in correct.values():
        corrects.append(v)
    incorrects = []  # list of incorrect predictions per classes
    for v in incorrect.values():
        incorrects.append(v)

    # ~ Plotting section
    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    # Define 2 bars inside the plot
    rects1 = ax.bar(x - width / 2, corrects, width, align="edge", label='Correctly Classified Digits')
    rects2 = ax.bar(x + width / 2, incorrects, width, align="edge", label='Incorrectly Classified Digits')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Classification results for ' + model_name + ' model')
    ax.set_xticks(x + 0.2)
    ax.set_facecolor('white')
    ax.set_xticklabels(labels)
    ax.legend()
    autolabel(rects1, ax)
    autolabel(rects2, ax)

    # Save classification report
    plt.savefig('./images/' + model_name.replace(" ", "") + '_class_report.png')
    plt.show()
