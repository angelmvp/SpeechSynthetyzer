from torch.utils.data.dataloader import DataLoader
import torch
class DataLoaderMVP(DataLoader):
    def __init__(self,dataset,batch_size):
        super().__init__(dataset,batch_size,shuffle=True,collate_fn=self.collate_fn)
    def collate_fn(self,batch):
        """
        Esta funcion se encarga de hacer el padding de las secuencias en un batch al mismo largo.
        Util para los datos de entrada y salida tienen longitudes variables.
        """
        x_batch, y_batch = zip(*batch)
        max_len = max(
            max(len(x) for x in x_batch),
            max(len(y) for y in y_batch)
        )

        # añadimos padding a ambas listas al mismo largo
        x_padded = torch.stack([
            torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)])
            for x in x_batch
        ])
        y_padded = torch.stack([
            torch.cat([y, torch.zeros(max_len - len(y), dtype=torch.long)])
            for y in y_batch
        ])

        return x_padded, y_padded