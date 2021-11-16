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
    #the "count" array and time array that keeps track of which star is running and how long it takes
    #I use the row index of each star in the input file as the 'count' instead of Gaia_source_ids 
    #because if the job crashes on MARCC,it is easier to find the starting point to restart the job
    count = []
    times = []
    for i in range(0,len):
        #The iloc function will get the row we are interested in running and keep the data structure
        start = time.time()
        #run isochrones
        iso_code.run_isochrones(data_input.iloc[[i]],name)
        end = time.time()
        #append the row index and time after a star has finished running, and write the result to time.csv
        count.append(i)
        times.append(end-start)
        time_df = pd.DataFrame({"count":count,"time":times})
        time_df.to_csv("time.csv",index=False)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
