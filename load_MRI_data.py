import os
import scipy.io
import numpy as np

rootdir = "/Users/marialsaker/MRImages/"
numpy_files_path = "/Users/marialsaker/git/pyfigures-master/MRI_data/"

for subdir, dirs, files in os.walk(rootdir):
    for dir in dirs:
        #print(os.path.join(subdir, dir))
        params = dir.split("_")
        if len(params)<5:
            continue
        name = f"{params[3]}_{params[2]}_{params[4]}"
        content = os.listdir(subdir+dir)
        path = subdir+dir+"/"
        if params[1] == "3Dcones":
            for fi in content:
                fi_split = fi.split(".")
                if fi_split[-1] == "mat":
                    path = path + fi
            mat = scipy.io.loadmat(path)
            im_volume = mat['bb']
            im_volume = np.transpose(im_volume, (2,0,1)) # (x, y, z) = (0, 1, 2)
            np.save(numpy_files_path+name, im_volume)
        if params[1] == "Optimizer":
            name = name[:-2] + "_optimizer"
            for fi in content:
                fi_split = fi.split(".")
                if fi_split[1] == "txt" and len(fi_split[0].split("_"))<4:
                    print(fi)
                    with open(numpy_files_path+name+".txt", "w") as wfile:
                        with open(path+fi, "r") as rfile:
                            for line in rfile:
                                wfile.write(line)