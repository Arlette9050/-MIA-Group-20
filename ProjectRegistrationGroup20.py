# import the needed modules
import sys
import registration_util as util

# get the other code files from the folder code 
sys.path.append("../code")

# get the image path of the orginal image
I_path = '../data/image_data/1_1_t1.tif'
# get the image path of the moving image
Im_path = '../data/image_data/1_1_t1_d.tif'
# get the points in the image with the help of the function
X, Xm = util.cpselect(I_path, Im_path)

# print the location of the points in the image
print('X:\n{}'.format(X))
print('Xm:\n{}'.format(Xm))