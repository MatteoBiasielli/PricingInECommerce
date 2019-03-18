import os
import pandas as pd
import pricing.data.answers_conversions_dictionaries as acd

RAWDATADIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
PREPROCDATADIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'preprocessed_data')


class DataPreprocesser:

    def __init__(self, path=None, no_basic_preprocessing=False):
        rawdata = pd.read_csv(os.path.join(RAWDATADIRPATH, 'rawdata.csv') if path is None else path)

        if not no_basic_preprocessing:
            # remove date and language
            rawdata.drop(rawdata.columns[[0, 1]], axis=1, inplace=True)

            # unify columns in english/italian
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
            rawdata.columns = ["Gender", "Age", "Location", "University", "Background", "Employed", "Outdoor",
                               "Max_Price"]

            # fix categorical data
            rawdata.University.replace(('Yes', 'No'), (1, 0), inplace=True)
            rawdata.Employed.replace(('Yes', 'No'), (1, 0), inplace=True)
            rawdata.Outdoor.replace(('Yes', 'No'), (1, 0), inplace=True)

            location_dummies = pd.get_dummies(rawdata['Location'], prefix='Location')
            rawdata.drop("Location", axis=1, inplace=True)
            background_dummies = pd.get_dummies(rawdata['Background'], prefix='Background')
            rawdata.drop("Background", axis=1, inplace=True)

            rawdata = pd.concat([rawdata, location_dummies, background_dummies], axis=1)

        self.data = rawdata

    def get_processed_data(self):
        return self.data

    def process_data(self, function_headers_list):
        for foo in function_headers_list:
            self.data = foo(self.data)

    def save_data(self, path=None):
        self.data.to_csv(os.path.join(PREPROCDATADIRPATH, 'processed_data.csv') if path is None else path, index=False)


if __name__ == '__main__':
    dp = DataPreprocesser()
    dp.save_data()
    # print(DataPreprocesser(path='./preprocessed_data/processed_data.csv', no_basic_preprocessing=True).data)
