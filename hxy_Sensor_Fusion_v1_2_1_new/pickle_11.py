import pickle

with open("lidar_test_result", "rb") as get_myprofile:
    while True:
        try:
            print (pickle.load(get_myprofile))
        except:
            break