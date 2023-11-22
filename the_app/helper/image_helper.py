import requests

# Download image from url and return as bytes
def download_image(url: str) -> bytes:
    """Download image from url and return as bytes"""
    # return null if url is not valid
    if url == "" or url is None or url == None:
        return None
    # return null if url is not an url
    if not url.startswith("http"):
        return None
    response = requests.get(url)
    return response.content