from PIL import Image

# Creating a image object, of the sample image
img = Image.open(r'sample.jpg')

# A 12-value tuple which is a transform matrix for dropping
# green channel (in this case)
matrix1 = ( 1, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 1, 0)

matrix2 = ( 0, 0, 0, 0,
           0, 0, 0, 0,
           0, 0, 1, 0)

# Transforming the image to RGB using the aforementioned matrix
#img = img.convert("RGB", matrix1)
img = img.convert("RGB", matrix2)

# Displaying the image
img.show()
