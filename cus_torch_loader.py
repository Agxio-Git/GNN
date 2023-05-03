import cv2
import pandas as pd
import os
import numpy as np
import torch
from torch_geometric.data import Dataset, download_url
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import torch
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt




  

class GNN_dataset_creator(Dataset):
    
    # Base url to download the files
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNN_dataset_creator, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return "Optimum_lesion_list_metadata_Grade_Invasive_Seg_oldnew.xlsx"

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt']

    def download(self):
        pass

    def add_grid_positions(self, Grid_size):
        horizontal_quad = []
        vertical_quad = []
        grid = {}
        count = 0
        posX = []
        posY = []
        for x_step in range(0, Grid_size):
            for y_step in range(0, Grid_size):
                grid_section = {}
                grid_section["x"] = x_step
                grid_section["y"] = y_step
                grid["{}".format(count)] = grid_section
                posX.append(x_step)
                posY.append(y_step)
                count += 1
                
                    
        posX = np.array(posX)
        posY = np.array(posY)
        posX = np.expand_dims(posX, axis = -1)
        posY = np.expand_dims(posY, axis = -1)
        NL = np.concatenate((posX, posY), axis=-1)
        
        return grid, NL
    
    def create_edge_and_position_information(self):
        Grid_size = 50
        NL_P = []
        NL_B = []
    
        create_grid, POS = self.add_grid_positions(Grid_size)
        list_keys = list(create_grid)
        for key1 in list_keys:
            for key2 in list_keys:
                if key1 != key2:
                    x1 = create_grid[key1]["x"]
                    y1 = create_grid[key1]["y"]
                    
                    x2 = create_grid[key2]["x"]
                    y2 = create_grid[key2]["y"]
                    if x1 - 1 <= x2 <= x1 + 1:
                        if y1 - 1 <= y2 <= y1 + 1:
                            NL_P.append(key1)
                            NL_B.append(key2)
                            
    
        NL_P = np.array(NL_P, dtype = int)
        NL_B = np.array(NL_B, dtype = int)
        NL_P = np.expand_dims(NL_P, axis=0)
        NL_B = np.expand_dims(NL_B, axis=0)
    
        NL = np.concatenate((NL_P, NL_B), axis=0)

        #edge_labels = ["source", "target"]
        #position_labels = ["x", "y"]
        #Edges = pd.DataFrame(data=NL, columns=edge_labels)
        #Positions = pd.DataFrame(data=POS, columns=position_labels)
        return  POS, NL
    def process(self):
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.
        
        path = os.path.join(self.raw_dir, "Optimum_lesion_list_metadata_Grade_Invasive_Seg_oldnew.xlsx")
        data = pd.read_excel(path)
        #data = data[data.Conspicuity == "Obvious"]
        #data = data[(data['MLO_Conspicuity'] == "Obvious") & (data['CC_Conspicuity'] == "Obvious")]
        data = data[(data['Conspicuity'] == "Obvious")]
        data = data[(data['DcisGrade'] != "NDL")]
        
        
        node_attrs = pd.read_excel(path)

        #node_attrs.index += 1
        
        #positions, edges = self.create_edge_and_position_information()
        
        
        
        
        
        data_list = []
        ids_list = list(str(node_attrs['C_MLO']))

        for g_idx in ids_list:
            node_ids = node_attrs.loc[node_attrs['C_MLO']==g_idx].index
            img_path = os.path.join("DATASET/All_images_Optimum/Segs", g_idx + ".png")
            mask_path = os.path.join("DATASET/All_images_Optimum/Segs", g_idx + ".png")
            # Node features
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (256, 256),interpolation=cv2.INTER_CUBIC)
            
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (256, 256),interpolation=cv2.INTER_CUBIC)
            
            attributes = node_attrs.iloc[node_ids, 4:]

            # Edges info
            
            
            # Graph label
            label = node_attrs.loc[node_ids, "High"]
            # Normalize the edges indices
            edge_idx = torch.tensor(edges, dtype=torch.long)
            pos_idx = torch.tensor(positions, dtype=torch.long)
            map_dict = {v.item():i for i,v in enumerate(torch.unique(edge_idx))}
            map_edge = torch.zeros_like(edge_idx)
            for k,v in map_dict.items():
                map_edge[edge_idx==k] = v
            
            # Convert the DataFrames into tensors 
            x = torch.tensor(attributes.to_numpy().transpose(), dtype=torch.float)
            #pad = torch.zeros((attrs.shape[0], 4), dtype=torch.float)
            #x = torch.cat((attrs, pad), dim=-1)
            
            #pad = torch.zeros((attrs.shape[0], 4), dtype=torch.float)
            #x = torch.cat((attrs, pad), dim=-1)

            edge_idx = map_edge.long()
            np_lab = label.to_numpy()
            y = torch.tensor(np_lab, dtype=torch.long)
            
            graph = Data(x=x, edge_index=edge_idx,  y=y, pos=pos_idx)
            
            data_list.append(graph)
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), osp.join(self.processed_dir, f'data_{idx}.pt'))  
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

dataset = GNN_dataset_creator(root="data")
print(dataset[0])

