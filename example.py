import imageio

img = imageio.imread('scene1.row3.col1.ppm')
img1 = imageio.imread('scene1.row3.col2.ppm')

print(img.shape, img1.shape)
