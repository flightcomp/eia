
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