import torch, torchaudio
from torch import nn
from torch.utils.data import DataLoader
from urbansounddataset import UrbanSoundDataset
from mycnn import CNNNet

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001

ANNOTATIONS_FILE =  "C:/tmp/sound_datasets/urbansound8k/metadata/UrbanSound8k.csv"
AUDIO_DIR = "C:/tmp/sound_datasets/urbansound8k/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(dataset=train_data, batch_size=batch_size)
    return train_data_loader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device) #annoying part I cant skip

        # Calculate Loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        
        # Backpropagate loss and update weights
        optimizer.zero_grad()   # resets the gradient, needed in pytorch
        loss.backward()         # backpropagation
        optimizer.step()        # update

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
    print("End of training")
         


if __name__ == "__main__":  
      
    if torch.cuda.is_available():
        device = "cuda"
    else: 
        device = "cpu"    
    print("Using device", device)
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    # instantiate dataset object and dataloader
    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)    
    train_dataloader = create_data_loader(usd,BATCH_SIZE)
    
    # construct model and assign to device
    cnn = CNNNet().to(device) # we need to assign the network to a device (CPU or GPU)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),lr=LEARNING_RATE)
    
    train(cnn,train_dataloader,loss_fn, optimizer, device, EPOCHS)
    
    torch.save(cnn.state_dict(), "cnn.pth")