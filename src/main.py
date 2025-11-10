from dataset.dataset import DatasetMVP
from vocab.vocab import VocabMVP
from dataloader.dataloader import DataLoaderMVP
def main():
	vocabulario = VocabMVP()
	dataset = DatasetMVP(vocabulario)
	print(len(dataset))
	print("Exitoso")
	
	# print(dataset.data[500:505])
	# print(dataset.data[500:505])
	print(dataset.__getitem__(100))
	BATCH_SIZE = 5000
	
	
	
	dataloader = DataLoaderMVP(dataset,batch_size=BATCH_SIZE)
	for x_batch,y_batch in dataloader:
		print(x_batch,y_batch)
main()

