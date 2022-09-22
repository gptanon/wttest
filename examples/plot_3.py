# -*- coding: utf-8 -*-
"""
Plot
====
Example
"""
# from wavespin.visuals import plot
# plot([1, 2, 3], title="Title")

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# # mpl.rcParams['savefig.bbox'] = 'tight'

# for j in range(2):
#     fig, axes = plt.subplots(4, 5, figsize=(11/2, 15.89/2))

#     for i, ax in enumerate(axes.flat):
#         cmap = 'bwr' if j == 0 else None
#         ax.imshow(np.random.randn(16, 16), cmap=cmap)
#         ax.set_xlabel(str(i), fontsize=20)
#         if i % 5 == 0:
#             ax.set_ylabel(str(i), fontsize=20)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         for spine in ax.spines:
#             ax.spines[spine].set_visible(False)


#     fig.suptitle("Title", weight='bold', fontsize=26, y=1.025)
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1,
#                         wspace=.02, hspace=.02)
#     fig.supxlabel("supXlabel", weight='bold', fontsize=24, y=-.05)
#     fig.supylabel("supYlabel", weight='bold', fontsize=24, x=-.066)

#     plt.suptitle("Title", weight='bold', fontsize=26, y=1.025)
#     # plt.savefig("yes.png")
#     plt.show()
