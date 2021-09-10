import test
import time
import iso_code
import pandas as pd

def main():
    name="rave"
    #I am reding the input data as a pandas dataframe because it handles large files much better
    #and saves memory. Also, it is faster. I set up the data type for source id to read the entire number as integer
    #and avoid truncating it
    start = time.time()
    data_input = pd.read_csv("isochrones_input/{}_isochrones.csv".format(name), dtype={'dr3_source_id':int})
    #Although standard pratice is to vectorize usage, in our case we do need to use a for loop.
    #Just a for loop to run it for all the stars in my file list
    len = data_input.shape[0]
    for i in range(0,1):
        #The iloc function will get the row we are interested in running and keep the data structure
        iso_code.run_isochrones(data_input.iloc[[i]],name)
        with open('count.txt', 'w') as file:
            file.write(str(i))
    # iso_code.read_isochrones(name)
    # iso_code.scatter_plot(name)
    # iso_code.check_parallax(name)
    end = time.time()

    print("\nTime: "+str(end-start))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
