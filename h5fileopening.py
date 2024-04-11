import h5py
from bs4 import BeautifulSoup
import configparser

configparser = configparser.RawConfigParser()

rootdir = "/Users/marialsaker/MRImages/Series42_3Dcones_2552_QuadratureCoil_X/"
filename = "ScanArchive_NO1011MR01_20240229_113134895.h5"


# def open_xml(pack, stringname):
#     lines = pack[stringname][()][0]#[0])
#     Bs_data = BeautifulSoup(lines, "xml")
#     with open("testing.txt", "w") as wfile:
#         wfile.write(Bs_data.prettify())

# with open("testing.txt", "w") as wfile:
#     with h5py.File(rootdir+filename, "r") as f:
#         # Print all root level object names (aka keys) 
#         # these can be group or dataset names 
#         print("Keys: %s" % f.keys())

#         # get first object name/key; may or may NOT be a group
#         items = f['Header.xml'][()]
#         print(f, 'OriginalHeader.xml')
#         open_xml(f, 'DownloadData.xml')
        #print(items)
        # for key in items.keys():
        #     print("Acquisition keys: ", items[key])
            #open_xml("StorageMetaData.xml")
            # for key2 in items[key]:
            #     print(type(items[key][key2][()]))
                # all = items[key][key2][()][0]
                # all_splitted = str(all).split("\\n\\n")
                # for i in all_splitted:
                #     wfile.write(i+"\n")



    # for a_group_key in f.keys():
    #     #a_group_key = list(f.keys())[0]
    #     #print(a_group_key)
    #     # get the object type for a_group_key: usually group or dataset
    #     # print(type(f[a_group_key])) 

    #     # # If a_group_key is a group name, 
    #     # # this gets the object names in the group and returns as a list
    #     # data = list(f[a_group_key])
    #     # print(data)

    #     # # If a_group_key is a dataset name, 
    #     # # this gets the dataset values and returns as a list
    #     # data = list(f[a_group_key])
    #     # preferred methods to get dataset values:
    #     #ds_obj = f[a_group_key]      # returns as a h5py dataset object
    #     #ds_arr = f[a_group_key][()]  # returns as a numpy array


f = h5py.File(rootdir+filename, "r")
f_list = list(f)
print(f_list)

print(f"Scan time 2552 readouts : {255.200//60:.0f}:{(255.200/60 - 255.200//60)*60:.0f}")
print(f"Scan time 2552 readouts : {140.200//60:.0f}:{(140.200/60 - 140.200//60)*60:.0f}")
print(f"Scan time 2552 readouts : {19.700//60:.0f}:{(19.700/60 - 19.700//60)*60:.0f}")
