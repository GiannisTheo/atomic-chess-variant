import os
import torch
from torch import nn
from copy import deepcopy
from EarlyStopping import EarlyStopping
import NeuralNetworks as nets 
from customdataset import CustomDataset, encode_piece, encode_move_2table, encode_value_table
from torch.utils.data import Dataset, DataLoader
import numpy as np
import joblib

# loads dataset and splits into train/val/test (80,10,10)
def train_val_test_split(path):
  train_chunks = []
  for f in os.listdir(path):  
      train_chunks.append(path+f)
  test_chunks = train_chunks[-4:]
  del train_chunks[-4:]
  valid_chunks = train_chunks[-4:]
  del train_chunks[-4:]
  return train_chunks, valid_chunks, test_chunks



def train_net(modelType, train_chunks, valid_chunks, trainType,policy_target, checkpoint_path, testing = False, modelPath = None):

  #initialize parameters
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  epochs = 15
  start_epoch = 0
  end_epoch = epochs
  model = deepcopy(modelType)
  criterion = nn.NLLLoss() if trainType == 'policy' else torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
  val_loss_min = np.Inf
  train_losses = []
  valid_losses = []
  average_train_losses = []
  average_valid_losses = []
  
  if(modelPath):
    checkpoint = torch.load(modelPath, map_location=device)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    average_train_losses = checkpoint['train_losses']
    average_valid_losses = checkpoint['val_losses']
    val_loss_min = checkpoint['val_loss_min']
    start_epoch = checkpoint['epoch'] + 1
    model.to(device)
    ESObject = EarlyStopping(val_loss_min, best=val_loss_min, path = checkpoint_path)
  else:
    model.to(device)
    ESObject = EarlyStopping(val_loss_min, path = checkpoint_path)
  model.to(device)
  
  # train model
  for epoch in range(start_epoch, end_epoch):
      model.train()
      if (trainType == 'policy') and (policy_target == 'head'): model.freezeBase() 
      for n, chunk in enumerate(train_chunks):
          f = open(chunk,'rb')
          table_chunk = joblib.load(f)
          f.close() 
          if(trainType == 'policy'):
            loader = DataLoader(CustomDataset(encode_move_2table(table_chunk),policy_target), 64)
          else:
            loader = DataLoader(CustomDataset(encode_value_table(table_chunk), 'base'), 64)
          ## loader is train_loader
          for s, p in loader:
              s=s.float().to(device)
              p=p.float().to(device)
              optimizer.zero_grad() 
              p0=model(s)
              loss = criterion(p0, torch.argmax(p.view(-1,64),dim = 1))  if trainType == 'policy' else criterion(p0.squeeze(1), p)
              train_losses.append(loss.item())
              loss.backward()
              optimizer.step()
            
          
          if(n % 10 == 0):
            print(f'epoch: {epoch:2}  batch: {n:4} loss: {train_losses[-1] :10.8f}') 
          if (testing == True):  
            break

      print("evaluating...")
      model.eval() # prep model for evaluation
      for n, chunk in enumerate(valid_chunks):
          f = open(chunk,'rb')
          table_chunk = joblib.load(f)
          f.close() 
          if(trainType == 'policy'):
            loader = DataLoader(CustomDataset(encode_move_2table(table_chunk),policy_target), 64)
          else:
            loader = DataLoader(CustomDataset(encode_value_table(table_chunk), 'base'), 64)

          
          for s, p in loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            s=s.float().to(device)
            p=p.float().to(device)
            p0 = model(s)
            # calculate the loss
            loss=criterion(p0, torch.argmax(p.view(-1,64),dim = 1)) if trainType == 'policy' else criterion(p0.squeeze(1), p)
            # record validation loss
            valid_losses.append(loss.item())
          if (testing == True):  
            break

      
      

      # clear lists to track next epoch
    
      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)
      average_train_losses.append(train_loss)
      average_valid_losses.append(valid_loss)
      train_losses = []
      valid_losses = []

      epoch_len = len(str(epochs))
      print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
              f'train_loss: {train_loss:.5f} ' +
              f'valid_loss: {valid_loss:.5f}')
      print(print_msg)

      ESObject(valid_loss, modelType, model, optimizer, epoch, average_train_losses, average_valid_losses)



def Visualize_Loss(train_loss,valid_loss):
  # visualize the loss as the network trained
  fig = plt.figure(figsize=(10,8))
  plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
  plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

  # find position of lowest validation loss
  minposs = valid_loss.index(min(valid_loss))+1 
  plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.ylim(0, 2) # consistent scale
  plt.xlim(0, len(train_loss)+1) # consistent scale
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.show()
  #completename = os.path.join('losses_curves',filename)
  #fig.savefig(completename, bbox_inches='tight')