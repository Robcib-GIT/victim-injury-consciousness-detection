def crop_function(image, r1, r2, c1, c2):
    h = len(image)
    w = len(image[0])

    if r1 < 0:
        r1 = 0
    if c1 < 0:
        c1 = 0
    if r2 > h:
        r2 = h
    if c2 > w:
        c2 = w

    cropped= image[r1:r2, c1:c2]

    return cropped

from cProfile import label
import cv2

TRIAGE_COLOR = {
    "green":  (0, 255, 0),
    "yellow": (0, 255, 255),
    "red":    (0, 0, 255),
    "black":  (0, 0, 0),
}


def draw_bbox(
    image,
    x1, y1, x2, y2,
    triage="green",
    thickness=2,
    label=None
):
    # renk
    color = TRIAGE_COLOR.get((triage or "").lower(), (0, 255, 0))

    # bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label and triage:
        text = f"{label} | {triage}"
    elif label:
        text = label
    else:
        text = ""

    if text:
        cv2.putText(
            image,
            text,
            (x1, max(y1 - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return image

if __name__ == "__main__":
    # test code
    img = cv2.imread("test.jpg")
    img = draw_bbox(img, 50, 50, 200, 300, triage="red", label="person_1")
    cv2.imwrite("test_with_box.jpg", img)