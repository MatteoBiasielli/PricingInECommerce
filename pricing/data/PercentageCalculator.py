import matplotlib.pyplot as plt
import numpy as np

from pricing.data.DataPreprocesser import DataPreprocesser as DP


class PercentageCalculator:

    def __init__(self, data) -> None:
        self.data = data

    def compare_columns(self, column_names_list, plot=True):
        # permit only comparisons between same types of features
        prefixes = [i.split('_', 1)[0] for i in column_names_list]
        assert prefixes.count(prefixes[0]) == len(prefixes), "The columns must be comparable."

        column_percentages = {}  # dictionary {"column_name": percentage}
        for column_name in column_names_list:
            column_percentages[column_name] = self.get_single_column_percentage(column_name, plot=False)[column_name]

        # if they don't add up to 100% -> rescale
        percentage_sum = sum(column_percentages.values())
        if percentage_sum != 100:
            for column_name, percentage in column_percentages.items():
                column_percentages[column_name] = round(((percentage * 100) / percentage_sum), 2)

        if plot:
            self.__plot(column_percentages)

        return column_percentages

    def get_single_column_percentage(self, column_name, plot=True):
        column = self.data[column_name].ravel()
        # computes percentage of ones in column
        percentage = round(((np.count_nonzero(column) / column.size) * 100), 2)  # two decimals rounding
        # both % of ones and zeros are returned
        percentages = {column_name: percentage, "No_" + column_name: 100 - percentage}

        if plot:
            self.__plot(percentages)

        return percentages

    def get_numerical_column_percentages(self, column_name, list_of_ranges):
        for rang in list_of_ranges:
            assert rang[0] <= rang[1], "The numeric ranges must be realistic."
        column = self.data[column_name].ravel()
        percentages = {}
        for rang in list_of_ranges:
            # count number of samples in the range and compute percentage on total
            percentage = round((sum(1 for n in column if rang[0] <= n <= rang[1]) / column.size) * 100, 2)
            percentages[rang] = percentage

        return percentages

    @staticmethod
    def __plot(columns_percentages):
        labels = columns_percentages.keys()
        sizes = columns_percentages.values()

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
        ax1.axis('equal')

        plt.show()


if __name__ == '__main__':
    all_people = DP(path='../data/preprocessed_data/processed_data.csv',
                    no_basic_preprocessing=True).get_processed_data()
    PC = PercentageCalculator(all_people)
    print(PC.get_numerical_column_percentages("Max_Price", [(0, 50), (0, 249), (250, 500)]))
    print(PC.compare_columns(["Background_Scientific", "Background_Humanistic", "Background_Artistic"]))
    print(PC.get_single_column_percentage("University"))
