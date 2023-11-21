import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MNIST:
    """
    MNIST dataset loader and visualizer
    
    Attributes
    ----------
    dataframe : pandas.DataFrame
        MNIST dataset in pandas.DataFrame format
    mnist_data : numpy.ndarray
        MNIST dataset in numpy.ndarray format
    labels : numpy.ndarray
        MNIST labels in numpy.ndarray format
    digits : numpy.ndarray
        MNIST digits in numpy.ndarray format
        
    Methods
    -------
    exportStatisticalAnalysis()
        Export statistical analysis of MNIST dataset to csv file
    printUnusedPixels()
        Print unused pixels
    get(selected_number)
        Get selected_numberth data
    getLabel(selected_number)
        Get selected_numberth label
    getDigit(selected_number)
        Get selected_numberth digit
    displayImage(selected_number)
        Display selected_numberth image
    plotClassDistribution()
        Plot class distribution
    plotClassPercentage()
        Plot class percentage
        
    """

    def __init__(self):
        """
        Description
        ----------
        Constructor of MNIST class

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.dataframe = pd.read_csv('data/mnist.csv')
        self.mnist_data = self.dataframe.values
        self.labels = self.mnist_data[:, 0]
        self.digits = self.mnist_data[:, 1:]
        print('MNIST data loaded successfully')

    def getSize(self):
        """
        Description
        ----------
        Get number of data

        Parameters
        ----------
        None

        Returns
        -------
        int
            Number of data
        """
        return self.dataframe.shape[0]
    def exportStatisticalAnalysis(self):
        """
        Description
        ----------
        Export statistical analysis of MNIST dataset to csv file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.dataframe.describe().to_csv('data/mnist_statistical_analysis.csv', index=True, header=True)

    def printUnusedPixels(self):
        """
        Description
        ----------
        Print unused pixels

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        zero_mean = []
        zero_std  = []

        for i in range(1, 785):
            hand = self.dataframe.iloc[:, i].describe()[['mean', 'std']]
            if hand['mean'] == 0.0:
                zero_mean.append(i)
            if hand['std'] == 0.0:
                zero_std.append(i)

        print('Pixels with zero mean: ', zero_mean)
        print('Pixels with zero std: ', zero_std)        

    def get(self, selected_number):
        """
        Description
        ----------
        Get selected_numberth data

        Parameters
        ----------
        selected_number : int
            Selected number
        
        Returns
        -------
        numpy.ndarray
            selected_numberth data
        """
        return self.mnist_data[selected_number]
    
    def getLabel(self, selected_number):
        """
        Description
        ----------
        Get selected_numberth label

        Parameters
        ----------
        selected_number : int
            Selected number
        
        Returns
        -------
        int
            selected_numberth label"""
        return self.get(selected_number)[0]
    
    def getDigit(self, selected_number):
        """
        Description
        ----------
        Get selected_numberth digit

        Parameters
        ----------
        selected_number : int
            Selected number
        
        Returns
        -------
        numpy.ndarray
            selected_numberth digit
        """
        return self.get(selected_number)[1:]
    
    def displayImage(self, selected_number):
        """
        Description
        ----------
        Display selected_numberth image

        Parameters
        ----------
        selected_number : int
            Selected number
        
        Returns
        -------
        None
        """
        img_size = 28
        plt.title('Label is {label}'.format(label=self.labels[selected_number]))
        plt.imshow(self.digits[selected_number].reshape(img_size, img_size))
        plt.show()

    def plotClassDistribution(self):
        """
        Description
        ----------
        Plot class distribution

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        unique_labels = np.unique(self.labels)
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

        for i, label in enumerate(unique_labels):
            plt.hist(self.labels[self.labels==label], bins=[label, label+1], color=colors[i], rwidth=0.8, align='left')

        plt.xlabel('Labels')
        plt.ylabel('Frequency')
        plt.title('Class Distribution')
        plt.xticks(range(10))
        plt.show()
    
    def plotClassPercentage(self):
        """
        Description
        ----------
        Plot class percentage

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        class_distribution = np.bincount(self.labels)
        class_labels = np.unique(self.labels)
        major_class = class_labels[np.argmax(class_distribution)]
        major_percentage = max(class_distribution) / sum(class_distribution) * 100

        plt.pie(class_distribution, labels=class_labels, autopct='%1.1f%%')
        plt.title('Class Distribution')
        plt.text(-0.3, -1.2, f'Majority class: {major_class} ({major_percentage:.1f}%)', fontsize=12)
        plt.show()


if __name__ == '__main__':
    mnist = MNIST()
    mnist.exportStatisticalAnalysis()
    mnist.printUnusedPixels()
    mnist.displayImage(0)
    mnist.plotClassDistribution()
    mnist.plotClassPercentage()