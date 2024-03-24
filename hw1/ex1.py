import cv2
import matplotlib.pyplot as plt


# Load an image from file as function
def load_image(image_path):
	"""
	Load an image from file, using OpenCV
	"""

	image = cv2.imread(image_path, cv2.IMREAD_COLOR)

	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	return rgb_image

# Display an image as function
def display_image(image, title="Image"):
	"""
	Display an image using matplotlib. Remember to use plt.show() to display the image
	"""

	plt.imshow(image)
	plt.title(title)
	plt.axis('off')

	plt.show()

# grayscale an image as function
def grayscale_image(image):
	"""
	Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
	3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
	[1] is given by:
		p = 0.299R + 0.587G + 0.114B
	Where the R, G, B are the values for each of the corresponding channels. We will do this by
	creating an array called img_gray with the same shape as img
	"""
	gray_img = image.copy()

	for i in range(gray_img.shape[0]):
		for j in range(gray_img.shape[1]):
			b, g, r = gray_img[i, j]
			gray_value = 0.299 * r + 0.587 * g + 0.11 * b
			gray_img[i, j] = gray_value

	return gray_img

# Save an image as function
def save_image(image, output_path):
	"""
	Save an image to file using OpenCV
	"""

	cv2.imwrite(output_path, image)

# flip an image as function
def flip_image(image):
	"""
	Flip an image horizontally using OpenCV
	"""
	flipped_image = cv2.flip(image, 1)
	return flipped_image

# rotate an image as function
def rotate_image(image, angle):
	"""
	Rotate an image using OpenCV. The angle is in degrees
	"""

	center = (image.shape[1] // 2, image.shape[0] // 2)
	rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
	return rotated_image


if __name__ == "__main__":
	# Load an image from file
	img = load_image("uet.png")

	# Display the image
	display_image(img, "Original Image")

	# Convert the image to grayscale
	img_gray = grayscale_image(img)

	# Display the grayscale image
	display_image(img_gray, "Grayscale Image")

	# Save the grayscale image
	save_image(img_gray, "images/lena_gray.jpg")

	# Flip the grayscale image
	img_gray_flipped = flip_image(img_gray)

	# Display the flipped grayscale image
	display_image(img_gray_flipped, "Flipped Grayscale Image")

	# Rotate the grayscale image
	img_gray_rotated = rotate_image(img_gray, 45)

	# Display the rotated grayscale image
	display_image(img_gray_rotated, "Rotated Grayscale Image")

	# Save the rotated grayscale image
	save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

	# Show the images
	plt.show()
