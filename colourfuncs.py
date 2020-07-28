import matplotlib as mpl


class MidpointNormalise(mpl.colors.Normalize):
    """
    Colour map normaliser that allows specification of a midpoint.
    From https://github.com/mwaskom/seaborn/issues/1309#issue-267483557
    """

    def __init__(self, vmin, vmax, midpoint=None, clip=False):
        #if vmin == vmax: self.degenerate = True
        #else: self.degenerate = False
        if midpoint == None: self.midpoint = (vmax + vmin) / 2
        else: self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        #if self.degenerate: return 'w'
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


# class HSVcmap:
#     """

#     """
    
#     def __init__(self, h, s, v, resolution=10):
#         """
#         Make a colour map that varies smoothly across one of the HSV dimensions,
#         while keeping the other two constant.
#         """
#         assert type(h) in (list, tuple) and type(s) in (list, tuple) and type(v) in (list, tuple), \
#                "h, s and v must be lists or tuples."
#         l = len(h)
#         assert l == len(s) == len(v), "h, s and v must have the same length."
#         # Assemble dictionary of RGB colours.
#         cdict = {"red":[], "green": [], "blue": []}
#         for i, (hi, si, vi) in enumerate(zip(h, s, v)):
#             pos = i / (l-1)
#             print(pos, hi, si, vi)

#             ri, gi, bi = mpl.colors.hsl_to_rgb([hi, si, vi])
#             print(ri, gi, bi)
#             print()



# cmap = HSVcmap(h=[0,0.5,0.75,1], s=[1,1,1,1], v=[1,1,1,1])



# A custom colour map, specified in HSV, then converted to RGB.
# cdict = {'red':   [[0.0,  1.0, 1.0],
#                    #[0.5,  0.25, 0.25],
#                    [0.5, 0.8, 0.8],
#                    [1.0,  0.0, 0.0]],
#          'green': [[0.0,  0.0, 0.0],
#                    #[0.5,  0.25, 0.25],
#                    [0.5, 0.6, 0.6],
#                    [1.0,  0.8, 0.8]],
#          'blue':  [[0.0,  0.0, 0.0],
#                    #[0.5,  1.0, 1.0],
#                    [0.5, 0.0, 0.0],
#                    [1.0,  0.0, 0.0]]}  

# hsv = ([]

# print(cdict)

# custom_cmap = mpl.colors.LinearSegmentedColormap('custom_cmap', segmentdata=cdict)