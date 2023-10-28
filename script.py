import function as fn
import numpy as np

fn.take_photo()

whiteImgs=fn.load_images_from_folder('./Training Images/White/')
#print(whiteImgs[1])

verti_line_mat = np.float32([[-1,2,-1],
                            [-1,2,-1],
                            [-1,2,-1]])

print(fn.lineDetector(whiteImgs[1],verti_line_mat))