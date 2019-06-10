import cv2 as cv

def get_saliency_fine_grained(image):
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    if success is False:
        raise Exception("Couldn't get saliency map")
    saliencyMap = (saliencyMap * 255).astype("uint8")
    return saliencyMap