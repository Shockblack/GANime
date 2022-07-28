#------------------------------------------------------------------------------
# Filename: mal_scrape.py
#
# Programmer: Aiden Zelakiewicz (https://github.com/Shockblack)
#
# Dependencies: requests, BeautifulSoup, PIL, os, io
# 
# Description:
#   Scrapes the MyAnimeList website for anime face images and information. Goes
#   download the list of images from the characters page to a specified directory.
#   If no directory is specified, the images will be saved to the 'data' directory
#   in the current working directory. If this directory does not exist, it will
#   be created.
#
#   The images are downloaded in jpg format. All website images that are not
#   character faces are skipped. This includes the mini banner, badge, challenge,
#   icon, and question mark images. The question mark images are for characters
#   that do not have a face image associated with them on MyAnimeList.
#
# Revision History:
#   27-Jul-2022:  File Created
# 
#------------------------------------------------------------------------------

# Adding all necessary imports
import requests
from bs4 import *
from PIL import Image
from io import BytesIO
import os

def download_images_from_url(url, directory, convert_to_jpg=True):
    """Downloads images from the given url and saves them to the specified directory.

    Parameters
    ----------
    url : str
        The url to download the images from.
    directory : str
        The directory to save the images to.
    convert_to_jpg : bool, optional
        Whether or not to convert the images to jpg. The default is True.
    """

    # Downloading the page
    page = requests.get(url)

    # Parsing the page
    soup = BeautifulSoup(page.text, 'html.parser')

    # Finding the images
    images = soup.find_all('img')

    # Looping through the images
    for image in images:

        # Getting the image url
        try:
            image_url = image['src']
        except KeyError:
            image_url = image['data-src']
        except:
            print("Error: Could not find image source.")

        if "http" not in image_url:
            continue

        # Getting the image name
        image_name = image_url.split('/')[-1].lower()

        # Update image name to be clean
        image_name = image_name.split('?')[0]

        # Images that are not part of the character list are not downloaded
        rejected_imgs = ["mini_banner", "badge", "challenge", "icon", "question"]

        # Skips the image if it is not a face image (question mark)
        if any(substring in image_name for substring in rejected_imgs):
            print(f"Image not available for {image_name}")
            continue

        # Downloading the image
        image_data = requests.get(image_url)

        if convert_to_jpg:
            # Converting the image to jpg
            image_data = Image.open(BytesIO(image_data.content))
            image_data = image_data.convert('RGB')
            image_data = image_data.save(directory + '/' + image_name, 'JPEG')
        else:
            # Saving the image
            image_data = image_data.content
            with open(directory + '/' + image_name, 'wb') as f:
                f.write(image_data)

        # Printing the image name
        print(image_name)

def parse_mal_characters(num_images, directory=None):
    """Parses the MyAnimeList characters page and downloads the images using
    the download_images_from_url function. Given a max number of characters to
    download, it will loop through all pages until it reaches the max number.

    Parameters
    ----------
    num_images : int
        The max number of characters to download.
    directory : str, optional
        The directory to save the images to. The default is None.
        If a default directory is not specified, the images will be saved to the
        'data' directory in the current working directory. If this directory
        does not exist, it will be created.
    """

    # If no directory is specified, set the directory to the default
    if type(directory) == type(None):

        # If the data directory does not exist, create it
        if os.path.exists('data/'):
            directory = 'data/'
        else:
            directory = 'data/'
            os.mkdir(directory)

    # Looping through the pages
    for page in range(0, num_images, 50):
        # Creating the url
        url = 'https://myanimelist.net/character.php?limit=' + str(page)
        # Downloading the images from the url
        download_images_from_url(url, directory=directory)

if __name__ == "__main__":
    parse_mal_characters(num_images=100)