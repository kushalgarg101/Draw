import torch
from torch.utils.data import DataLoader,Dataset

class DataPrepare(Dataset):
    def __init__(self, Dataset):
        super().__init__()

        self.data = Dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        sample_x = self.data[idx][0]
        sample_y = self.data[idx][1]

        sample_x = torch.from_numpy(sample_x).permute(2, 0, 1).float()
        sample_y = torch.tensor(sample_y)

        return sample_x,sample_y
    
def dataloader(data, batch_size=32, shuffle=True, num_workers=0):

    dataset = DataPrepare(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader