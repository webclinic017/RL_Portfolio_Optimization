import pickle


def pickle_out_data(data, pickle_name):
    pickle_out = open(pickle_name + ".pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()


def pickle_in_data(pickle_name):
    pickle_in = open(pickle_name + ".pickle", "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()

    return data