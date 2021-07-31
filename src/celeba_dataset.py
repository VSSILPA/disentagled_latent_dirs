from PIL import Image
from torch.utils.data import Dataset
import os
from natsort import natsorted


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
    Args:
      root_dir (string): Directory with all the images
      transform (callable, optional): transform to be applied to each image sample
    """
        # Read names of images in the root directory
        image_names = os.listdir(root_dir)

        self.root_dir = root_dir
        self.transform = transform
        self.image_names = natsorted(image_names)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the path to the image
        img_path = os.path.join(self.root_dir, self.image_names[idx])
        # Load image and convert it to RGB
        img = Image.open(img_path).convert('RGB')
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)

        return img
