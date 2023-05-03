import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import warnings
import pandas as pd
import argparse
from cus_torch_loader import GNN_dataset_creator

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
parser.add_argument('--embed_dims', type=int,default=64, help='input patch size of network input')

args = parser.parse_args()
print(args)


#data = MNISTSuperpixels(root=".")# Load the MNISTSuperpixel dataset

#data_loader structure root/data/raw/morisita_test.xlsx
data = GNN_dataset_creator(root="data")
print(data[0])
#print(data.num_classes)
### I need the histology data in this format

embedding_size = args.embed_dims
class GCN(torch.nn.Module):
    def __init__(self):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)
        # GCN layers
        self.initial_conv = GCNConv(data.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.conv3 = GCNConv(embedding_size, embedding_size)
        # Output layer
        self.out = Linear(embedding_size*2, data.num_classes)
    def forward(self, x, edge_index, batch_index):
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.tanh(hidden)
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([gmp(hidden, batch_index), 
                            gap(hidden, batch_index)], dim=1)
        # Apply a final (linear) classifier.
        out = self.out(hidden)
        return out, hidden

model = GCN()



data_size = len(data)
#data=data[:int(data_size * 0.2)]

warnings.filterwarnings("ignore")

# Cross EntrophyLoss
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  
# Use GPU for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# Wrap data in a data loader
data_size = len(data)
loader = DataLoader(data[:int(data_size * 0.8)], 
                    batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(data[int(data_size * 0.8):], 
                         batch_size=args.batch_size, shuffle=True)

def train(data):
    # Enumerate over the data
    for batch in loader:
      # Use GPU
      batch.to(device)  
      # Reset gradients

      optimizer.zero_grad() 
      # Passing the node features and the connection info
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      # Calculating the loss and gradients
      loss = torch.sqrt(loss_fn(pred, batch.y))       
      loss.backward()  
      # Update using the gradients
      optimizer.step()   
    return loss, embedding
print("Starting training...")
losses = []
for epoch in range(args.max_epochs):
    loss, h = train(data)
    losses.append(loss)
    if epoch % 10 == 0:
      print(f"Epoch {epoch} | Train Loss {loss}")
test_batch = next(iter(test_loader))
with torch.no_grad():
    test_batch.to(device)
    pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
    pred=torch.argmax(pred,dim=1)
    print(test_batch.y[0])#Actual REsult
    print(pred[0])#Predicted Result
    print(test_batch.y[1])#Actual REsult
    print(pred[1])#Predicted Result
    print(test_batch.y[2])#Actual REsult
    print(pred[2])#Predicted Result
    print(test_batch.y[3])#Actual REsult
    print(pred[3])#Predicted Result
    print(test_batch.y[4])#Actual REsult
    print(pred[4])#Predicted Result
    print(test_batch.y[5])#Actual REsult
    print(pred[5])#Predicted Result
