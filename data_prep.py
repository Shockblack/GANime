#--------------------------------------------------------------
# File: data_prep.py
#
# Programmer: Aiden Zelakiewicz (zelakiewicz.1@osu.edu)
#
# Dependencies: pytorch, torchvision, skimage, os
#
# Description:
#   Contains the data preparation, creating the dataloader.
#
# Revision History:
#   25-Nov-2022:  File Created
#--------------------------------------------------------------

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from skimage import io
import os

class AnimeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        
        for image in os.listdir(self.root_dir):
            if image.endswith(".jpg"):
                img = io.imread(os.path.join(self.root_dir, image), pilmode='RGB', plugin='imageio')
                self.images.append(img)


    def __len__(self):
        # Very compact way to check amount of files in directory
        return len([entry for entry in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, entry))])

    def __getitem__(self, idx):
        # Getting the image
        # img_name = os.path.join(self.root_dir, os.listdir(self.root_dir)[idx])
        # try:
        #     img = io.imread(img_name, pilmode='RGB', plugin='imageio')
        # except:
        #     print("Error reading image: ", img_name)
        #     return None
        img = self.images[idx]
        # Performing the transformation
        if type(self.transform) != type(None):
            img = self.transform(img)

            # Performing the cropping
            img = transforms.functional.crop(img, 5, 0, 100, 100)
            img = transforms.functional.resize(img, (64,64))
        
        return img



def get_dataloader(batch_size, root_dir, num_workers=0):
    # Creating the dataset
    dataset = AnimeDataset(root_dir, transform=transforms.Compose([transforms.ToTensor()]))

    # Creating the dataloader
    if num_workers == 0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, multiprocessing_context='fork')

    return train_loader