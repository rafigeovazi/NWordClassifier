import cv2
import numpy as np
import pytesseract
from PIL import ImageGrab

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def detect_and_highlight_N(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = pytesseract.image_to_boxes(rgb)

    h, w, _ = image.shape

    for b in boxes.splitlines():
        b = b.split()
        char = b[0]
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])

        y = h - y
        y2 = h - y2

        if char.upper() == "N":
            cv2.rectangle(image, (x, y2), (x2, y), (255, 255, 255), -1)
            cv2.putText(image, "N", (x - 10, y - 5), cv2.FONT_HERSHEY_COMPLEX, 2.5, (0, 0, 0), 5)

    return image

while True:
    img = np.array(cv2.imread("TextImage.png"))
    processed_img = detect_and_highlight_N(img)

    cv2.imshow("Text Detection (Highlight N)", processed_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
