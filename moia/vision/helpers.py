
from urllib.parse import urlparse

def read_image(image_url):
    from PIL import Image, UnidentifiedImageError
    import requests
    from io import BytesIO
    try:
        if uri_validator(image_url):
            header = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36"}
            response = requests.get(image_url, headers=header)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
            else:
                print(f"Failed to download image from URL: {image_url}. Status code: {response.status_code}")
                return None
        else:
            # No es una url sino un path
            image = Image.open(image_url)
    except UnidentifiedImageError:
        print(f"Unable to identify image format for URL: {image_url}")
        return None
    return image

def read_image_with_flags(image_url, flags):
    '''
    IMREAD_UNCHANGED: -1, If set, return the loaded image as is (with alpha channel, otherwise it gets cropped).
    IMREAD_GRAYSCALE:  0, If set, always convert image to the single channel grayscale image.
    IMREAD_COLOR_BGR:  1, If set, always convert image to the 3 channel BGR color image.
    IMREAD_COLOR:      1, Same as previous
    IMREAD_ANYDEPTH:   2, If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
    IMREAD_ANYCOLOR:   4, If set, the image is read in any possible color format. 
    '''
    import urllib.request, cv2, numpy
    if uri_validator(image_url):
        # Leer la imagen desde la URL
        request = urllib.request.Request(image_url)
        request.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36')
        with urllib.request.urlopen(request) as response:
            image = response.read()
        # Convertir la imagen a un array NumPy
        i = numpy.asarray(bytearray(image), dtype="uint8")
        # i = numpy.frombuffer(image, numpy.uint8)
        image = cv2.imdecode(i, flags)
    else:
        # No es una url sino un path
        image = cv2.imread(image_url, flags)
    return image

def uri_validator(x):
    try:
        result = urlparse(x)
        return all([result.scheme, result.netloc])
    except AttributeError:
        return False
        
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python import vision
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.


    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()
