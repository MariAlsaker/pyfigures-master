from matplotlib import pyplot as plt
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
from smithplot import SmithAxes


fig = plt.figure(figsize=[5, 5])
ax = fig.add_subplot(1,1,1, projection="smith")

reals=0
ims = 0
vals_s11 = (reals + ims * 1j)
plots = ax.plot(vals_s11, markevery=1, datatype=SmithAxes.S_PARAMETER, markersize=1)
plots[0].set(color="r")
plt.savefig(f'smith_chart.png', transparent=True, dpi=300)
plt.show()


