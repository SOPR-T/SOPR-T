import numpy as np
import torch
import torch.nn as nn

class RankTransformer(nn.Module):
    def __init__(self, num_enc_opt, num_features_rank, num_features_enc, num_seg, using_cls):
        super(RankTransformer, self).__init__()
        self.num_enc_opt = num_enc_opt
        self.num_features_rank = num_features_rank
        self.num_seg = num_seg
        self.usingCLS = using_cls

        self.model = nn.Sequential(
            nn.Linear(self.num_features_rank, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(num_features_enc, self.num_enc_opt),
            nn.BatchNorm1d(self.num_enc_opt),
            nn.ReLU(),
        )
        self.cls_inp = torch.nn.Parameter(torch.FloatTensor(self.num_enc_opt))
        self.encoder_lowlev = nn.Sequential(
            nn.Linear(self.num_features_rank, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, self.num_features_rank),

            nn.Linear(self.num_features_rank, 128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(128, self.num_features_rank),
        )
        self.encoder_hilev = nn.Sequential(
            nn.Linear(self.num_features_rank, 512),
            nn.Dropout(0.1),
            nn.ReLU(),                                        
            nn.Linear(512, self.num_features_rank),

            nn.Linear(self.num_features_rank, 512),
            nn.Dropout(0.1),
            nn.ReLU(),                                        
            nn.Linear(512, self.num_features_rank),

            nn.Linear(self.num_features_rank, 512),
            nn.Dropout(0.1),
            nn.ReLU(),                                        
            nn.Linear(512, self.num_features_rank),

            nn.Linear(self.num_features_rank, 512),
            nn.Dropout(0.1),
            nn.ReLU(),                                        
            nn.Linear(512, self.num_features_rank),

            nn.Linear(self.num_features_rank, 512),
            nn.Dropout(0.1),
            nn.ReLU(),                                        
            nn.Linear(512, self.num_features_rank),

            nn.Linear(self.num_features_rank, 512),
            nn.Dropout(0.1),
            nn.ReLU(),                                        
            nn.Linear(512, self.num_features_rank),
        )
        self.fc_score = nn.Sequential(
            nn.Linear(self.num_seg, 1),
        )

    def forward(self, x, s_clus_size, mode='mean', scorer=True):
        x = self.fc(x)  # increase dim using single_layer fc, (P*S, E)
        num_state = sum(s_clus_size)
        num_po = int(x.shape[0]/num_state)
        if type(s_clus_size) is np.ndarray:
            num_seg = s_clus_size.shape[0]
        else:
            num_seg = len(s_clus_size)
        x = x.view(int(num_state), int(num_po*self.num_enc_opt))    # (S, P*E)
        
        # Low-level Transformer encoder
        ind_start = 0
        for i in range(num_seg):
            ind_end = ind_start+ s_clus_size[i]
            enc_out_clus = x[int(ind_start): int(ind_end),:]
            enc_out_clus = torch.reshape(enc_out_clus, (enc_out_clus.shape[0], int(enc_out_clus.shape[1]/self.num_enc_opt),self.num_enc_opt))  # (N_c, P, E)
            # Low-lev TE
            enc_out_clus = self.encoder_lowlev(enc_out_clus)
        
            # Pooling
            enc_out_clus = torch.mean(enc_out_clus, 0)  # (P, E)
            if i == 0:
                z_pool = enc_out_clus
            else:
                z_pool = torch.cat([z_pool, enc_out_clus], dim = 0)
            ind_start = ind_end

        del enc_out_clus
        del x

        # Add CLS's embedding
        if self.usingCLS:
            cls = self.cls_inp.repeat(num_po).reshape(1, -1)
            z_pool = torch.cat([cls, z_pool], dim=0)

            # Reshape as [S, N, E] for Transformer encoder
            z_pool = torch.reshape(z_pool, (int(num_seg)+1, num_po, int(self.num_enc_opt)))
        else:
            z_pool = torch.reshape(z_pool, (int(num_seg), num_po, int(self.num_enc_opt)))

        # High-level Transformer encoder
        z_pool = self.encoder_hilev(z_pool)

        # Form and Select policy pairs
        if self.usingCLS:
            z_pool = z_pool[0]
            z_pool = z_pool.mean(1).unsqueeze(-1)
        else:
            # Score
            z_pool = self.model(z_pool).transpose(0,1)   #  shape=(N, S, 1)
            z_pool = torch.squeeze(z_pool)  #  shape=(N, S)
            if mode == 'mean':
                z_pool = z_pool.mean(1).unsqueeze(-1) #self.fc_score(z_pool)
            elif mode == 'sort':
                z_pool = self.fc_score(torch.sort(z_pool, dim=1)[0])
        return z_pool
