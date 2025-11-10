from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
class DataLoaderMVP(DataLoader):
    def __init__(self,dataset,batch_size):
        super().__init__(dataset,batch_size,shuffle=True,collate_fn=self.collate_fn)
    def collate_fn(self,batch):
        x_batch, y_batch = zip(*batch)
        x_padded = pad_sequence(x_batch, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y_batch, batch_first=True, padding_value=0)
        return x_padded, y_padded