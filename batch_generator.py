import tensorflow as tf
import tensorflow.keras as keras
"""

None of these work yet


"""
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)



#CSV reader
def csv_feature_generator(x_path,y_path, batch_size=32):
	# open the input file for reading
    f1 = open(x_path, "r")
    f2 = open(y_path, "r")
	# loop indefinitely
    while True:
        # initialize our batch of data and labels
        data = []
        labels = []
        # keep looping until we reach our batch size
        while len(data) < batch_size:
            # attempt to read the next row of the CSV file
            row = f1.readline()
            if row == "":
                f1.seek(0)
                row = f1.readline()
            row = row.strip().split(",")
            features = np.array(row, dtype="float")
            # update the data and label lists
            data.append(features)

        while len(labels)<batch_size:
            row = f2.readline()
            if row == "":
                f2.seek(0)
                row = f2.readline()
            row = row.strip().split(",")
            label = np.array(row,dtype='float')
            labels.append(label)

        # yield the batch to the calling function
        yield (np.array(data), np.array(labels))

def csv_data_generator(x_path, batch_size=32):
	# open the input file for reading
    f1 = open(x_path, "r")
	# loop indefinitely
    while True:
        # initialize our batch of data and labels
        data = []
        labels = []
        # keep looping until we reach our batch size
        while len(data) < batch_size:
            # attempt to read the next row of the CSV file
            row = f1.readline()
            if row == "":
                f1.seek(0)
                row = f1.readline()
            row = row.strip().split(",")
            features = np.array(row, dtype="float")
            # update the data and label lists
            data.append(features)

        # yield the batch to the calling function
        yield np.array(data)