import numpy as np
import cv2


class PhotoRecognizer:
    def __init__(self, matrixImage):
        self.matrixImage = matrixImage
        self.distribution, size = self.distributionOfFrequency()
        self.blackAndWhite = self.blackAndWhite()
        self.reverseImage()
        darknessLevel = (self.pictureAnalytics(size)) ** 2
        self.matrixImage = self.roll()
        print(darknessLevel)
        self.matrixImage = self.increase_contrast_gamma_correction(gamma=darknessLevel)
        self.matrixImage = self.denoise(self.matrixImage, tv_weight=darknessLevel * 15)[0]
        self.matrixImage = self.binarizationApproximation()
        self.lastDistribution, _ = self.distributionOfFrequency()
        self.matrixImage = self.binarization(size)
        self.afterBinDenoise()
        # self.focusDenoise()
        # kernelSize = self.kernelSelection()
        # print(kernelSize)
        # self.matrixImage = self.opening(1, 1, kernelSize, kernelSize) * 255
        # # self.matrixImage = self.closing(1, 1, kernelSize, kernelSize)*255
        # self.matrixImage = self.histeq(2)[0]
        # self.matrixImage = self.binDemo(self.matrixImage)

    def roundOdd(self, number):
        rounded = round(number)
        if rounded % 2 == 0: return rounded - 1
        return rounded

    def distributionOfFrequency(self):
        distribution = {}
        for row in self.matrixImage:
            for element in row:
                distribution[element] = distribution[element] + 1 if (element in distribution) else 1
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True)), len(self.matrixImage) * len(
            self.matrixImage[0])

    def pictureAnalytics(self, amount):
        darknessLevel, normalization = 0, 100 / 25500
        for value in self.distribution:
            darknessLevel += (self.distribution.get(value) / amount) * value
        return (1 - darknessLevel * normalization) * 3

    def blackAndWhite(self):
        maxFrequencyPixel, maxPixel, minPixel = max(self.distribution, key=self.distribution.get), np.max(
            self.matrixImage), np.min(self.matrixImage)
        return True if maxPixel - maxFrequencyPixel < maxFrequencyPixel - minPixel else False

    def reverseImage(self):
        if not self.blackAndWhite: self.matrixImage = 255 - self.matrixImage

    def increase_contrast_gamma_correction(self, gamma=1.5):
        # Створюємо таблицю відображення для гамма-корекції
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # Застосовуємо гамма-корекцію
        contrast_img = cv2.LUT(self.matrixImage, table)

        return contrast_img

    def histeq(self, nbr_bins=256):
        imhist, bins = np.histogram(self.matrixImage.flatten(), nbr_bins, density=True)
        cdf = imhist.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize
        im2 = np.interp(self.matrixImage.flatten(), bins[:-1],
                        cdf)  # use linear interpolation of cdf to find new pixel values
        return im2.reshape(self.matrixImage.shape), cdf

    def roll(self):
        kernel = np.array([[0.0, 0.0, -0.2, 0.0, 0.0],
                           [0.0, -0.2, -0.5, -0.2, 0.0],
                           [-0.2, -0.5, 5.0, -0.5, -0.2],
                           [0.0, -0.2, -0.5, -0.2, 0.0],
                           [0.0, 0.0, -0.2, 0.0, 0.0]])
        return cv2.filter2D(self.matrixImage, -1, kernel)

    def denoise(self, InitializeU, tolerance=0.1, tau=0.25, tv_weight=100):
        m, n = self.matrixImage.shape  # size of noisy image
        U, error = InitializeU, 1  # initialize
        Px, Py = np.zeros((m, n)), np.zeros((m, n))  # x,y-componentto the dual field
        while error > tolerance:
            oldU = U
            # gradient of primal variable
            GradUx, GradUy = np.roll(U, -1, axis=1) - U, np.roll(U, -1, axis=0) - U  # x,y-component of U's gradient

            # update the dual varible
            PxNew, PyNew = Px + (tau / tv_weight) * GradUx, Py + (
                    tau / tv_weight) * GradUy  # non-normalized update of x,y-component (dual)
            NormNew = np.maximum(1, np.sqrt(PxNew ** 2 + PyNew ** 2))
            Px, Py = PxNew / NormNew, PyNew / NormNew  # update of x,y-component (dual)

            # update the primal variable
            RxPx, RyPy = np.roll(Px, 1, axis=1), np.roll(Py, 1, axis=0)  # right x,y-translation of x-component
            DivP = (Px - RxPx) + (Py - RyPy)  # divergence of the dual field.
            U = self.matrixImage + tv_weight * DivP  # update of the primal variable

            # update of error
            error = np.linalg.norm(U - oldU) / np.sqrt(n * m)
        return U, self.matrixImage - U  # denoised image and texture residual

    def erode(self, kernelSize, iterations=1, image=[]):
        kernel = np.ones((kernelSize, kernelSize), dtype=np.uint8)
        output = self.matrixImage.copy() / 255 if len(image) == 0 else image
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2
        for _ in range(iterations):
            padded_image = np.pad(output, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                                  constant_values=1)
            temp_output = np.zeros_like(output)
            for i in range(pad_height, padded_image.shape[0] - pad_height):
                for j in range(pad_width, padded_image.shape[1] - pad_width):
                    region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
                    temp_output[i - pad_height, j - pad_width] = np.min(region * kernel)
            output = temp_output
        return output

    def dilate(self, kernelSize, iterations=1, image=[]):
        kernel = np.ones((kernelSize, kernelSize), dtype=np.uint8)
        output = self.matrixImage.copy() / 255 if len(image) == 0 else image
        pad_height = kernel.shape[0] // 2
        pad_width = kernel.shape[1] // 2
        for _ in range(iterations):
            padded_image = np.pad(output, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant',
                                  constant_values=0)
            temp_output = np.zeros_like(output)
            for i in range(pad_height, padded_image.shape[0] - pad_height):
                for j in range(pad_width, padded_image.shape[1] - pad_width):
                    region = padded_image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
                    temp_output[i - pad_height, j - pad_width] = np.max(region * kernel)
            output = temp_output
        return output

    def opening(self, iterationErode, iterationDilate, kernelErodeSize, kernelDilateSize):
        eroded = self.erode(kernelErodeSize, iterationErode)
        opened = self.dilate(kernelDilateSize, iterationDilate, image=eroded)
        return opened

    def closing(self, iterationErode, iterationDilate, kernelErodeSize, kernelDilateSize):
        dilated = self.dilate(kernelDilateSize, iterationDilate)
        closed = self.erode(kernelErodeSize, iterationErode, image=dilated)
        return closed

    def min_distance_between_contours(self, contour1, contour2):
        min_dist = float('inf')  # Ініціалізуємо мінімальну відстань нескінченністю

        for point1 in contour1:
            for point2 in contour2:
                dist = np.linalg.norm(point1 - point2)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def kernelSelection(self):
        img = self.matrixImage

        if img.dtype != np.uint8:  # Якщо зображення не у форматі 8-бітних цілих чисел
            img = img.astype(np.uint8)  # Конвертуємо у формат 8-бітних цілих чисел
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Оцінка середнього розміру об'єктів
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) == 0:
            areas = 1
        else:
            areas.remove(max(areas))
        avg_area = np.mean(areas)
        minArea = min(areas)

        # Знаходимо мінімальну відстань між усіма парами контурів
        num_contours = len(contours)
        if num_contours < 2:
            return 1  # Повертаємо kernel, якщо контурів менше двох

        min_dist = float('inf')
        for i in range(num_contours):
            for j in range(i + 1, num_contours):
                dist = self.min_distance_between_contours(contours[i], contours[j])
                if dist < min_dist:
                    min_dist = dist

        print(f"Мінімальна відстань між контурами: {min_dist}")
        avg_size = min(int(np.sqrt((minArea + avg_area) / 2)), min_dist)
        # avg_size = int(np.sqrt((0 + avg_area) / 2))
        kernel = max(1, self.roundOdd(avg_size))  # Збільшуємо ядро до найближчого непарного числа
        return kernel

    def binarizationApproximation(self):
        maxPixel, minPixel = np.max(self.matrixImage), np.min(self.matrixImage)
        return np.array([[(round(element / 100) * 100 if (
                    minPixel <= element <= (maxPixel - minPixel) / 2) else np.ceil(element / 100) * 100) for element in
                          row] for row in self.matrixImage])

    def binDemo(self, img):
        minPixel = np.min(img)
        return np.array([[(0 if (element == minPixel) else 255) for element in row] for row in img])

    def binarization(self, size):
        toBlack = [0]
        exists = list(self.lastDistribution.keys())
        for i in range(len(exists)):
            if self.lastDistribution[exists[i]] < size * 0.1: toBlack.append(exists[i])
        return np.array([[(0 if (element in toBlack) else 255) for element in row] for row in self.matrixImage])

    def afterBinDenoise(self):
        if self.matrixImage.dtype != np.uint8:  # Якщо зображення не у форматі 8-бітних цілих чисел
            self.matrixImage = self.matrixImage.astype(np.uint8)  # Конвертуємо у формат 8-бітних цілих чисел
        contours, _ = cv2.findContours(self.matrixImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Оцінка середнього розміру об'єктів
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) == 0:
            areas = 1
        else:
            areas.remove(max(areas))
        areaMax = max(areas)
        areaMin = min(areas)

        min_area_threshold = np.ceil(0.075 * areaMax)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_threshold:
                self.matrixImage = cv2.drawContours(self.matrixImage, [contour], -1, (255,), thickness=cv2.FILLED)

    def focusDenoise(self):
        if self.matrixImage.dtype != np.uint8:  # Якщо зображення не у форматі 8-бітних цілих чисел
            self.matrixImage = self.matrixImage.astype(np.uint8)  # Конвертуємо у формат 8-бітних цілих чисел
        contours, _ = cv2.findContours(self.matrixImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Оцінка середнього розміру об'єктів
        areas = [cv2.contourArea(c) for c in contours]
        if len(areas) == 0: return 1
        areas.sort()
        if areas[0] == 0.0: areas = areas[areas != 0.0]

        return 0
