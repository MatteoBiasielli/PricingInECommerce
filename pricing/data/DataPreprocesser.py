import os
import pandas as pd
import pricing.data.answers_conversions_dictionaries as acd


RAWDATADIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
PREPROCDATADIRPATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'preprocessed_data')


class DataPreprocesser:

    def __init__(self, path=None):
        rawdata = pd.read_csv(os.path.join(RAWDATADIRPATH, 'rawdata.csv') if path is None else path)
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
        print(rawdata)
        print(list(rawdata))
        self.data = rawdata

    def get_processed_data(self):
        return self.data

    def process_data(self, function_headers_list):
        for foo in function_headers_list:
            self.data = foo(self.data)

    def save_data(self, path=None):
        self.data.to_csv(os.path.join(PREPROCDATADIRPATH, 'processed_data.csv') if path is None else path)


if __name__ == '__main__':
    DataPreprocesser().save_data()