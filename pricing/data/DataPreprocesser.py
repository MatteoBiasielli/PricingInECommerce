import os

import pandas as pd

import pricing.data.answers_conversions_dictionaries as acd

RAWDATADIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
PREPROCDATADIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'preprocessed_data')


class DataPreprocesser:

    def __init__(self, path=None, no_basic_preprocessing=False, columns_to_remove=None):
        rawdata = pd.read_csv(os.path.join(RAWDATADIRPATH, 'rawdata.csv') if path is None else path)

        if not no_basic_preprocessing:
            # remove useless columns
            if columns_to_remove is not None:
                rawdata.drop(rawdata.columns[columns_to_remove], axis=1, inplace=True)

            # unify columns
            convs = acd.anwswers_translations()
            for c in convs:
                rawdata = rawdata.replace(to_replace=c[0], value=c[1])
            col2col = acd.question_to_question()
            nans = rawdata.isna()
            for c2c in col2col:
                for i in range(rawdata.shape[0]):
                    if nans[c2c[1]][i]:
                        rawdata.set_value(index=i, col=c2c[1], value=rawdata[c2c[0]][i])
            for c2c in col2col:
                rawdata = rawdata.drop(labels=[c2c[0]], axis=1)

            # simplify column names
            for column in rawdata:
                rawdata.rename(columns={column: col2col[next(i for i, lst in enumerate(col2col) if column in lst)][2]},
                               inplace=True)

            # fix categorical data
            for c in convs:
                # more readable names
                rawdata = rawdata.replace(to_replace=c[1], value=c[2])

            rawdata.replace(('Yes', 'No'), (1, 0), inplace=True)

            for column in self.__get_categorical_columns(rawdata):
                dummies = pd.get_dummies(rawdata[column], prefix=column)
                rawdata.drop(column, axis=1, inplace=True)
                rawdata = pd.concat([rawdata, dummies], axis=1)

        self.data = rawdata

    def get_processed_data(self):
        return self.data

    def process_data(self, function_headers_list):
        for foo in function_headers_list:
            self.data = foo(self.data)

    def save_data(self, path=None):
        self.data.to_csv(os.path.join(PREPROCDATADIRPATH, 'processed_data.csv') if path is None else path, index=False)

    def __get_categorical_columns(self, dataframe):
        cols = dataframe.columns
        num_cols = dataframe._get_numeric_data().columns
        return list(set(cols) - set(num_cols))


if __name__ == '__main__':
    # dp = DataPreprocesser(columns_to_remove=[0, 1])
    # dp.save_data()
    print(DataPreprocesser(path='./preprocessed_data/processed_data.csv', no_basic_preprocessing=True).data)
