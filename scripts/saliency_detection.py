import cv2 as cv

image = cv.imread("D:\\Image Matting Dataset\\merged_dataset\\1-1252426161dfXY_0.png")
saliency = cv.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)
saliencyMap = (saliencyMap * 255).astype("uint8")

threshMap = cv.threshold(saliencyMap.astype("uint8"), 0, 255,
	cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

cv.imshow("Image", image)
cv.imshow("Output", saliencyMap)
cv.imshow("Thresh", threshMap)

cv.waitKey(0)