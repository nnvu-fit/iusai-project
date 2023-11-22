import requests

# Download a file from a GitHub release and return the content
def download_file_from_github_release(url: str) -> bytes:
    """Download a file from a GitHub release and return the content"""
    # return null if url is not valid
    if url == "" or url is None or url == None:
        return None
    # return null if url is not an url
    if not url.startswith("http"):
        return None
    # download file
    response = requests.get(url)
    return response.content

# Classify image and return the result
def classify_image(image: bytes, model: bytes) -> str:
    """Classify image and return the result"""
    # return null if image is not valid
    if image == "" or image is None or image == None:
        return None
    # return null if model is not valid
    if model == "" or model is None or model == None:
        return None
    # classify image
    return "Classified!"