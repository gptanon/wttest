# -*- coding: utf-8 -*-
"""
Plot 4
======
Yes
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# class Animator1D(animation.TimedAnimation):
#     def __init__(self):
#         fig, ax = plt.subplots(1, 1)
#         self.lines = [ax.plot(np.random.randn(4))[0]]

#         animation.TimedAnimation.__init__(self, fig, blit=True)

#     def _draw_frame(self, frame_idx):
#         self.lines[0].set_data(np.arange(4), np.random.randn(4) / 5)
#         self._drawn_artists = [*self.lines]

#     def new_frame_seq(self):
#         return iter(range(30))

#     def _init_draw(self):
#         pass


# ani = Animator1D()
# ani.save('yes.mp4', fps=10, savefig_kwargs=dict(pad_inches=0))
