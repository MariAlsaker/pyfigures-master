import numpy as np
from matplotlib.colors import to_hex
from stl import mesh

# Save all your calculations here and use them in other files.
# Example: plane_at, loop induction calc, conductance_calc, bin_to_hex, trace_from_stl, ...

def plane_at(slice="x=00", extent=50):
    xx = np.linspace(-extent, extent, 101)
    zz = np.linspace(-extent, extent, 101)
    yy = np.linspace(-extent, extent, 101)
    if slice[0]=="x":
        Y, Z = np.meshgrid(yy, zz)
        X = np.ones_like(Y)*int(slice[-2:])
    elif slice[0]=="y":
        X, Z = np.meshgrid(xx, zz)
        Y = np.ones_like(X)*int(slice[-2:])
    elif slice[0]=="z":
        X, Y = np.meshgrid(xx, yy)
        Z = np.ones_like(X)*int(slice[-2:])
    return X, Y, Z

""" DISPLAY STL AND CREATE CURRENT LINES UNDERNEATH """
def bin_color_to_hex(x):
    """
    transform binary rgb into hex color
    source: https://magpylib.readthedocs.io/en/latest/_pages/docu/docu_graphics.html
    """
    sb = f"{x:015b}"[::-1]
    r = int(sb[:5], base=2) / 31
    g = int(sb[5:10], base=2) / 31
    b = int(sb[10:15], base=2) / 31
    return to_hex((r, g, b))

def trace_from_stl(stl_file, x_coords_current=None):
    """
    Generates a Magpylib 3D model trace dictionary from an *.stl file.
    backend: 'matplotlib' or 'plotly'
    source: https://magpylib.readthedocs.io/en/latest/_pages/docu/docu_graphics.html
    """
    # load stl file
    stl_mesh = mesh.Mesh.from_file(stl_file)

    # extract vertices and triangulation
    p, q, r = stl_mesh.vectors.shape
    vertices, ixr = np.unique(
        stl_mesh.vectors.reshape(p * q, r), return_inverse=True, axis=0
    )
    i = np.take(ixr, [3 * k for k in range(p)])
    j = np.take(ixr, [3 * k + 1 for k in range(p)])
    k = np.take(ixr, [3 * k + 2 for k in range(p)])
    x, y, z = vertices.T

    # generate and return a generic trace which can be translated into any backend
    colors = stl_mesh.attr.flatten()
    facecolor = np.array([bin_color_to_hex(c) for c in colors]).T
    trace = {
        "backend": "generic",
        "constructor": "mesh3d",
        "kwargs": dict(x=x, y=y, z=z, i=i, j=j, k=k, facecolor=facecolor),
    }
    if len(x_coords_current) > 0:
        # Extract coordinates of current line along x-axis
        idx = []
        for x1 in x_coords_current:
            idx_x = np.where(np.round(x, 0)==x1)[0]
            if len(idx_x)>0:
                ys_to_see = y[idx_x]
                #print(ys_to_see)
                idx_y = np.where(ys_to_see == 0.)[0]
                if len(idx_y)>0:
                    for i in idx_x[idx_y]:
                        idx.append(i)
        return trace, vertices[idx]
    return trace