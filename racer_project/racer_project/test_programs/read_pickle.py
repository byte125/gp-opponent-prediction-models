import pickle

# Open the file in binary read mode
with open('/home/sd/barc_data/testcurve.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can use the data object as it was originally saved
print(data)
