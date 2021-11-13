from os import times
import time
import iso_code
import pandas as pd
import numpy as np

def main():
    name="lamost"
    #I am reding the input data as a pandas dataframe because it handles large files much better
    #and saves memory. Also, it is faster. I set up the data type for source id to read the entire number as integer
    #and avoid truncating it
    data_input = pd.read_csv("isochrones_input/{}_isochrones.csv".format(name), dtype={'dr3_source_id':int})
    #Although standard pratice is to vectorize usage, in our case we do need to use a for loop.
    #Just a for loop to run it for all the stars in my file list
    len = data_input.shape[0]
    count = []
    times = []
    for i in range(6134,len):
        #The iloc function will get the row we are interested in running and keep the data structure
        start = time.time()
        try:
            iso_code.run_isochrones(data_input.iloc[[i]],name)
        except ZeroDivisionError:
            continue
        # with open('count.txt', 'w') as file:
        #     file.write(str(i))
        end = time.time()
        count.append(i)
        times.append(end-start)
        time_df = pd.DataFrame({"count":count,"time":times})
        time_df.to_csv("time.csv",index=False)
    # iso_code.read_isochrones(name)
    # iso_code.scatter_plot(name)
    # iso_code.check_parallax(name)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
