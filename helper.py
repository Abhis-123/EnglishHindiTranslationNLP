import pickle

# saving
def save_variable(variable, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(variable, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
def load_variable(file_path):
    with open(file_path, 'rb') as handle:
        variable = pickle.load(handle)
    return variable