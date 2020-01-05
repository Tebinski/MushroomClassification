"""
This code download the files of kaggle dataset
https://www.kaggle.com/uciml/mushroom-classification
"""
import os
import kaggle

if __name__ == '__main__':

    if not os.path.exists("./data/mushrooms.csv"):
        kaggle.api.authenticate()
        path = r'./data/'
        kaggle.api.dataset_download_files('uciml/mushroom-classification',
                                           path=path,
                                           unzip=True)
        print(f'The dataset has been downloaded at {path}')
    else:
        print('Data is already downloaded')
