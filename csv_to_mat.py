import sys
import os
import csv
import pandas
import scipy.io

IC_READY = 'images/ic_go_search_api_holo_light.png'
IC_FINISHED = 'images/ic_checkmark_holo_light.png'
IC_ERROR = 'images/ic_dialog_alert_holo_light.png'

def convert_file(csvpath):
    matpath = csvpath[:-4] + '.mat'
    matpath = matpath.replace("csv_data", "mat_data")
    print(matpath)
    # read first line of csv file to get column names and replace ' ' with '_'
    r = csv.reader(open(csvpath))
    header = next(r)
    names = [x.replace(' ', '_') for x in header]

    # load the data into a pandas dataframe using the appropriate names
    df = pandas.read_csv(csvpath, names=names, header=0, skiprows=1, skip_footer=1)

    # write the dataframe to a mat file
    df_dict = {c: list(df[c]) for c in df.columns}
    scipy.io.savemat(matpath, df_dict, oned_as='column')


def main():
    if len(sys.argv) > 1:
        for f in sys.argv[1:]:
            #print ("Converting: ", f)
            convert_file(f)
            print ("Done")

if __name__ == '__main__':
    main()
