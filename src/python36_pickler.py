import pickle
from import_1dify import stretchArray

vid = pickle.load(open("smaller_movie_batched_diff_framesplit.p", "rb"), encoding="latin1")

pickle.dump(vid[1200], open("sparse_vickie_movement.p", "wb"))
