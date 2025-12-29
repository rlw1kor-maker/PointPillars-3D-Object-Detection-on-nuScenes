import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

class NuScenesDataset:
    def __init__(self, dataroot):
        self.nusc = NuScenes(
            version='v1.0-mini',
            dataroot=dataroot,
            verbose=True
        )
        self.samples = self.nusc.sample

    def __len__(self):
        return len(self.samples)

    def get_lidar(self, idx):
        sample = self.samples[idx]
        token = sample['data']['LIDAR_TOP']
        path = self.nusc.get_sample_data_path(token)
        pc = LidarPointCloud.from_file(path)
        return pc.points.T  # (N,4)
