import os
import scipy.io
import numpy as np
from datetime import datetime as dt

rootdir = "/Users/marialsaker/MRImages/"
numpy_files_path = "/Users/marialsaker/git/pyfigures-master/MRI_data/"
saved_info = []

for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        #print(os.path.join(subdir, dir))
        params = dir.split("_")
        if len(params)<5:
            continue
        nm = f"{params[3]}_{params[2]}_{params[4]}"
        content = os.listdir(subdir+dir)
        path = subdir+dir+"/"
        if params[1] == "3Dcones":
            for fi in content:
                fi_split = fi.split(".")
                if fi_split[-1] == "mat":
                    path = path + fi
            mat = scipy.io.loadmat(path)
            scan_time =  mat['h']['image'][0][0][0][0]["im_actual_dt"][0]#mat['h']['rdb_hdr'][0][0][0][0]["scan_time"][0]
            routs = mat['h']['image'][0][0][0][0]["sctime"][0]
            saved_info.append((f"{params[1]}_{params[2]}_{params[3]}_{params[4]}", routs, scan_time))
            im_volume = mat['bb']
            im_volume = np.transpose(im_volume, (2,0,1)) # (x, y, z) = (0, 1, 2)
#             #np.save(numpy_files_path+name, im_volume)
#         if params[1] == "Optimizer":
#             nm = nm[:-2] + "_optimizer"
#             for fi in content:
#                 fi_split = fi.split(".")
#                 if fi_split[1] == "txt" and len(fi_split[0].split("_"))<4:
#                     with open(numpy_files_path+nm+".txt", "w") as wfile:
#                         with open(path+fi, "r") as rfile:
#                             for line in rfile:
#                                 wfile.write(line)

# for nm, rs, sct in saved_info:
#     with open(numpy_files_path+f"{nm}_X_optimizer"+".txt", "a") as wfile:
#         wfile.write(f"{rs} readouts -> {sct} min scan time\n")
times = []
for inf in saved_info:
    times.append(dt.fromtimestamp(int(inf[-1][0])))
np.array(times)
times = np.sort(times)
#print(times)
diffs = []
for i in range(len(times)-1):
    #print(saved_info[i][0])
    d = (times[i+1]-times[i])
    diffs.append([saved_info[i+1][0], d])
diffs = np.array(diffs)
diffs = diffs[diffs[:,1].argsort()]
print(diffs)

print(dt.fromtimestamp(170920629))