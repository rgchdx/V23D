import cv2


def sharpness_laplacian(frame_bgr) -> float:
    # Compute the variance of the Laplacian of the grayscale image as a sharpness socre.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def keep_frame_by_blur(score: float, threshold: float) -> bool:
    # only keep frames with sharpness score above the threshold
    return score >= threshold
