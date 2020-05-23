import os
import sys

import numpy as np
import yaml
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.modules.activation as activation

class Basset(nn.Module):
    def __init__(self, weight_path=None, activate_padding=False):
        """Initilize Basset Model

        Args:
            weight_path (str): weight path to give to have pretraining ( Default : None - No pretraining).
        """
        super(Basset, self).__init__()
        self.activate_padding=activate_padding
        # Block 1 :
        self.c1 = nn.Conv1d(4, 300, 19, padding=9 if self.activate_padding else 0)
        self.r1 = activation.ReLU()
        self.mp1 = nn.MaxPool1d(3, 3)  # kernel size, stride
        self.bn1 = nn.BatchNorm1d(300)

        # Block 2 :
        self.c2 = nn.Conv1d(300, 200, 11, padding=5 if self.activate_padding else 0)
        self.r2 = activation.ReLU()
        self.mp2 = nn.MaxPool1d(4, 4)
        self.bn2 = nn.BatchNorm1d(200)

        # Block 3 :
        self.c3 = nn.Conv1d(200, 200, 7, padding=4 if self.activate_padding else 0)
        self.r3 = activation.ReLU()
        self.mp3 = nn.MaxPool1d(4, 4)
        self.bn3 = nn.BatchNorm1d(200)

        # Block 4 : Fully Connected 1 :
        self.d4 = nn.Linear(2000, 1000) # 2000 : 600bp in
        self.bn4 = nn.BatchNorm1d(1000) # , 1e-05, 0.1, True
        self.r4 = activation.ReLU()
        self.dr4 = nn.Dropout(0.3)

        # Block 5 : Fully Connected 2 :
        self.d5 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000) #, 1e-05, 0.1, True
        self.r5 = activation.ReLU()
        self.dr5 = nn.Dropout(0.3)

        # Block 6 : Fully connected 3 :
        self.d6 = nn.Linear(1000, 164)
        if weight_path:
            self.load_weights(weight_path)

    def forward(self, x, embeddings=False, output=True):
        """Run the forward step

        Args:
            x (torch.tensor): Input of the model
            embeddings (type):if True forward return embeddings along with the output ( Default : False )
            output (type): if True return ( and compute the output ) ( Default : True )

        Returns:
            outputs and/or embeddings depending of the parameters

        """

        # Block 1
        x = self.bn1(self.mp1(self.r1(self.c1(x))))

        # Block 2
        x = self.bn2(self.mp2(self.r2(self.c2(x))))

        # Block 3
        em = self.bn3(self.mp3(self.r3(self.c3(x))))

        if output :
            # Flatten
            o = torch.flatten(em, start_dim=1)
            # FC1
            o = self.dr4(self.r4(self.bn4(self.d4(o))))

            # FC2
            o = self.dr5(self.r5(self.bn5(self.d5(o))))

            # FC3
            o = self.d6(o)
        if embeddings and output: return o, em
        if embeddings and not output : return em
        if not embeddings and output: return o

    def extract_embeddings(self, x, layer) :
        """Extract embeddings from the requested layer

        Args:
            x (torch.tensor): input
            layers (int): layer to extract embeddings from

        Returns:
            type: embedding of the requested layer
        """
        # Block 1
        x = self.r1(self.c1(x))
        if layer == 10 : return x

        x = self.bn1(self.mp1(x))
        if layer == 1 : return x

        # Block 2
        x = self.bn2(self.mp2(self.r2(self.c2(x))))
        if layer == 2 : return x

        # Block 3
        x = self.bn3(self.mp3(self.r3(self.c3(x))))
        if layer == 3 : return x

        x = torch.flatten(x, start_dim=1)
        if layer == 4 : return x

        # FC1
        x = self.dr4(self.r4(self.bn4(self.d4(x))))
        if layer == 5 : return x

        # FC2
        x = self.dr5(self.r5(self.bn5(self.d5(x))))
        if layer == 6 : return x

        # FC3
        x = self.d6(x)
        raise Exception('Wrong layer number choosen')


    def load_weights(self, weight_path):
        """Load the weights for the model.

        Args:
            weight_path (str): Path of the file containing the weigts

        """
        sd = torch.load(weight_path)
        new_dict = OrderedDict()
        keys = list(self.state_dict().keys())
        values = list(sd.values())
        for i in range(len(values)):
            v = values[i]
            if v.dim() > 1 :
                if v.shape[-1] ==1 :
                    new_dict[keys[i]] = v.squeeze(-1)
                    continue
            new_dict[keys[i]] = v
        self.load_state_dict(new_dict)


if __name__ == '__main__':
    # # Prepare and predict Basset 
    # dataloaders, targets = load_datas(constants['dataset_path'])
    # dt = iter(dataloaders['test'])
    # seq, l = next(dt)
    # seq = seq.permute(0, 1, 3, 2).to(constants['device'])
    #    
    # model = Basset(weight_path=constants['weight_path'])
    # model.eval()
    # model.to(device)
    # oe = model(seq.squeeze(-1))
    # score = torch.sigmoid(oe) # No final activation in basset

    # Load data
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    species = 'Human'
    x = np.load('../data/' + species + '_Data/one_hot_seqs_ACGT.npy') 
    # TODO: Pad zeros to both ends to satisfy 600 bp
    window_size = 600  
    x = np.zeros((x.shape[0], x.shape[1], window_size))
    x = x.astype(np.float32)
    print(x.shape)
    y = np.load('../data/' + species + '_Data/cell_type_array.npy')
    y = y.astype(np.float32)
    peak_names = np.load('../data/' + species + '_Data/peak_names.npy')

    batch_size = 4
    _, test_data, _, test_labels, _, test_names = train_test_split(x, y, peak_names, test_size=0.01, random_state=42)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Prepare and predict Basset
    weight_path = '/home/chendi/projects/XDNN/kipoi/example/basset_pretrained_kipoi_weights.pth'
    model = Basset(weight_path=weight_path)
    model.to(device)

    num_classes = 164
    predictions = torch.zeros(0, num_classes)
    model.eval() # Set model to evaluate mode
    with torch.no_grad():        
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            pred = model(seqs)
            pred = torch.sigmoid(pred)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
    predictions = predictions.numpy()
    
    # # Prepare and predict kipoi
    # import kipoi
    # num_classes = 164
    # predictions = torch.zeros(0, num_classes)

    # model_kipoi = kipoi.get_model('Basset').model
    # model_kipoi.to(device)
    # model_kipoi.eval()
    # with torch.no_grad():        
    #     for seqs, labels in test_loader:
    #         seqs = seqs.to(device)
    #         pred = model_kipoi(seqs)
    #         predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
    # predictions = predictions.numpy()
    
