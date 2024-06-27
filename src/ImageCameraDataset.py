from torch.utils.data import Dataset

class RobotImageDataset(Dataset):
    def __init__(self, images, cameras, silhouette):
        self.images = images
        self.cameras = cameras
        self.silhouette = silhouette

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        camera = self.cameras[idx]
        silhouette = self.silhouette[idx]
        return image, silhouette, camera