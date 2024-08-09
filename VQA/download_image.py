import requests
from PIL import Image
from io import BytesIO

# URL of the MRI image
image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b2/MRI_of_Human_Brain.jpg"  # Replace with actual image URL
response = requests.get(image_url)

# Check if the request was successful
if response.status_code == 200:
    image = Image.open(BytesIO(response.content))
    image.save("./brain_mri_image.png")
    print("Image downloaded and saved as 'brain_mri_image.png'")
else:
    print("Failed to download image. Status code:", response.status_code)
