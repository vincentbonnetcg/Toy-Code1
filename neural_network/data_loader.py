"""
@author: Vincent Bonnet
@description : Class to load the MNIST database into numpy array without high-level libraries
"""

import os
from urllib import request

class MNIST_Loader:
    '''
    Loader to download the MNIST database
    '''
    def __init__(self):
        self.filenames = [
            ["training_images", "train-images-idx3-ubyte.gz"],
            ["test_images", "t10k-images-idx3-ubyte.gz"],
            ["training_labels", "train-labels-idx1-ubyte.gz"],
            ["test_labels", "t10k-labels-idx1-ubyte.gz"]
            ]
        self.url = "http://yann.lecun.com/exdb/mnist/"
        self.download_folder = "mnist_data/"

    def download(self, force=False):
        '''
        Download filenames into specified folder
        '''
        # create the download folder
        os.makedirs(os.path.dirname(self.download_folder), exist_ok=True)

        # download the files
        num_filenames = len(self.filenames)
        for index, filename in enumerate(self.filenames):
            full_url = self.url + filename[1]
            full_path = self.download_folder + filename[1]

            print("=> Process file [{0}/{1}]".format(index + 1, num_filenames))
            if os.path.exists(full_path) and force is True:
                print(full_url + " already downloaded")
            else:
                print("downloading " + full_url + "...")
                request.urlretrieve(full_url, full_path)
                print("download complete.")

loader = MNIST_Loader()
loader.download(True)
