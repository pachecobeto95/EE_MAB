import cv2


class ImageProcessor(object):
	def __init__(self, distortion_type, distortion_lvl):
		self.distortion_type = distortion_type
		self.distortion_lvl = distortion_lvl

	def blur(self):
		self.dist_img = cv2.GaussianBlur(self.image, (4*self.distortion_lvl+1, 4*self.distortion_lvl+1), 
			self.distortion_lvl)

	def noise(self):
		noise = np.random.normal(0, self.distortion_lvl, self.image.shape).astype(np.uint8)
		self.dist_img = cv2.add(self.image, noise)

	def distortion_not_found():
		raise ValueError("Invalid distortion type. Please choose 'blur' or 'noise'.")

	def apply(self, image_path):
		self.image = cv2.imread(image_path)
		dist_name = getattr(self, self.distortion_type, self.distortion_not_found)
		dist_name()

	def save_distorted_image(self, output_path):
		cv2.imwrite(output_path, self.dist_img)

