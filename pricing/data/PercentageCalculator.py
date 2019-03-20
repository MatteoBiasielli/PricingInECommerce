import matplotlib.pyplot as plt
import numpy as np

from pricing.data.DataPreprocesser import DataPreprocesser as DP


class PercentageCalculator:

    def __init__(self, data) -> None:
        self.data = data

    def compare_percentages(self, column_names_list, plot=True):
        # permit only comparisons between same types of features
        prefixes = [i.split('_', 1)[0] for i in column_names_list]
        assert prefixes.count(prefixes[0]) == len(prefixes), "The columns must be comparable."

        column_percentages = {}  # dictionary {"column_name": percentage}
        for column_name in column_names_list:
            column_percentages[column_name] = self.compute_percentage(column_name)

        if plot:
            self.__plot(column_percentages)

        return column_percentages

    def compute_percentage(self, column_name):
        column = self.data[column_name].ravel()
        return round(((np.count_nonzero(column) / column.size) * 100), 2)  # two decimals rounding

    def __plot(self, columns_percentages):
        labels = columns_percentages.keys()
        sizes = columns_percentages.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')

        plt.show()


if __name__ == '__main__':
    all_people = DP(path='../data/preprocessed_data/processed_data.csv',
                    no_basic_preprocessing=True).get_processed_data()
    PC = PercentageCalculator(all_people)
    print(PC.compare_percentages(["Background_Scientific", "Background_Humanistic", "Background_Artistic"]))
    print(PC.compute_percentage("University"))
