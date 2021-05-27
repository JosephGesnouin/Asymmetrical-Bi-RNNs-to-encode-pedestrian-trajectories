import math
import torch
import torch.nn as nn
import numpy as np

class SocialNCE():
    '''
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    '''
    def __init__(self, obs_length, pred_length, head_projection, encoder_sample, temperature, horizon,sampling):

        # problem setting
        self.obs_length = obs_length
        self.pred_length = pred_length

        # nce models
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample

        # nce loss
        self.criterion = nn.CrossEntropyLoss()

        # nce param
        self.temperature = temperature
        self.horizon = horizon

        # sampling param
        self.noise_local = 0.1
        self.min_seperation = 0.2
        self.agent_zone = self.min_seperation * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])

    def spatial(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            Input:
                batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
                batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
                batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
            Output:
                loss: social nce loss
        '''
     
        surr=len(self.agent_zone)
        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------
        '''
        for i in range(batch_split.shape[0] - 1):
            traj_primary = batch_scene[:, batch_split[i]] # [time, 2]
            traj_neighbor = batch_scene[:, batch_split[i]+1:batch_split[i+1]] # [time, num, 2]
            plot_scene(traj_primary, traj_neighbor, fname='scene_{:d}.png'.format(i))
        '''
        
        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------
        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split)
        
        nan_detect=torch.isnan(sample_neg)
        sample_neg[nan_detect]=0
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        emb_obsv= self.head_projection(batch_feat[:,batch_split[:-1]])

        emb_pos= self.encoder_sample(sample_pos[-1].cuda())
        emb_neg= self.encoder_sample(sample_neg[-1].cuda())

        query= nn.functional.normalize(emb_obsv,dim=-1)

        key_pos= nn.functional.normalize(emb_pos,dim=-1)
        key_neg= nn.functional.normalize(emb_neg,dim=-1)
        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        sim_pos=(query*key_pos[None,:,:]).sum(dim=-1)
        sim_neg=(query[:,:,None,:]*key_neg[None,:,:]).sum(dim=-1)

        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
        loss=torch.zeros(batch_split.shape[0] - 1)
        for i in range(batch_split.shape[0] - 1):
            logits=torch.cat([sim_pos[:,i,None], sim_neg[:,i,(batch_split[i]-i)*surr:(batch_split[i+1]-i-1)*surr ] ],dim=1)/self.temperature
            labels=torch.zeros(logits.size(0),dtype=torch.long)
            logits = logits.cuda()
            labels = labels.cuda()
            loss[i]=(self.criterion(logits,labels))
           
        loss=torch.mean(loss)

        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        '''
        surr=len(self.agent_zone)


        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split)

        nan_detect=torch.isnan(sample_neg)
        sample_neg[nan_detect]=0

        selec_pos=sample_pos[1:self.horizon+1,:]
        selec_neg=sample_neg[1:self.horizon+1,:]

        time_pos = (torch.ones(selec_pos.size(1))[None, :] * (torch.arange(self.horizon) - (
            self.horizon-1.0)*(0.5))[:, None]) / self.horizon

        time_neg = (torch.ones(selec_neg.size(1))[None, :] * (torch.arange(self.horizon) - (
            self.horizon-1.0)*(0.5))[:, None]) / self.horizon
        
    
        emb_obsv= self.head_projection(batch_feat[:self.horizon,batch_split[:-1]])
        emb_pos= self.encoder_sample(selec_pos,time_pos[:,:,None])
        emb_neg= self.encoder_sample(selec_neg,time_neg[:,:,None])


        query= nn.functional.normalize(emb_obsv,dim=-1)
        
        key_pos= nn.functional.normalize(emb_pos,dim=-1)
        
        key_neg= nn.functional.normalize(emb_neg,dim=-1)
        

        sim_pos=(query*key_pos).sum(dim=-1)
      
        sim_neg=(query[:,:,None,:]*key_neg[:,None,:,:]).sum(dim=-1)
        
        loss=torch.zeros(batch_split.shape[0] - 1)

        for i in range(batch_split.shape[0] - 1):
            logits=torch.cat([sim_pos[:,i,None], sim_neg[:,i,(batch_split[i]-i)*surr:(batch_split[i+1]-i-1)*surr ] ],dim=1)/self.temperature
            labels=torch.zeros(logits.size(0),dtype=torch.long)
            loss[i]=(self.criterion(logits,labels))
           
        loss=torch.mean(loss)

        return loss

    def _sampling_spatial(self, batch_scene, batch_split):
        '''
        Input:
            batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
            batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
        Output:
            batch_samples_pos: trajectories close to the ground truth, tensor of shape [pred_length, nb primary agents]
            batch_samples_neg: trajectories ending in discomfort zones of other pedestrians, tensor of shape [pred_length, number of agent_zones * nb non primary agents in the batch]
        Description:
            batch_scene is a combination of scenes delimited by indices in batch_split
            Each index in batch_split starts a new scene with a primary pedestrian and various neighbour pedestrians
            Each pedestrian has obs_length+pred_length x,y positions
            The function returns augmented predicted trajectories of the primary pedestrian of each scene;
            batch_samples_positive: ones close to ground truth and batch_samples_negative: several ones to avoid
        '''

        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length]  
        ''' .detach().clone()''' #* DEBUG CODE *#
        # #####################################################
        #           TODO: fill the following code
        # #####################################################
        batch_samples_positive = None # list containing positive samples of all scenes
        batch_samples_negative = None # list containing negative samples of all scenes

        nb_coords = gt_future.shape[2]
        nb_zones  = self.agent_zone.shape[0]

        # introduce random perturbations to all ground truth trajectories to prevent overfitting and add to data variety
        # in the paper self.noise_local is set to 0.05 [m], for example
        for i in range(batch_scene.shape[1]):
            # generate random noise
            noise = np.random.multivariate_normal(np.zeros(nb_coords), self.noise_local*np.eye(nb_coords), self.pred_length)
            gt_future[:, i] = gt_future[:, i].detach().cpu() + noise

        # iterate over all batches
        for i in range(batch_split.shape[0] - 1):
            primary_pedestrian   = gt_future[:, batch_split[i]]
            neighbor_pedestrians = gt_future[:, batch_split[i]+1:batch_split[i+1]]

            nb_primary = 1 # dimension ignored during extraction of primary_pedestrian as it is 1
            nb_neighbors = neighbor_pedestrians.shape[1]
            # -----------------------------------------------------
            #                  Positive Samples
            # -----------------------------------------------------
            # ground truth trajectory with some additive noise creates the positive sample

            # noisy trajectory reshape to (pred_length, nb_primary, nb_coords)
            near_perfect_trajectory = (primary_pedestrian.detach().cpu() + noise).reshape(self.pred_length, nb_primary, nb_coords)
            # append scene's positive sample to batch
            if batch_samples_positive is None:
                batch_samples_positive = near_perfect_trajectory
            else:
                batch_samples_positive = torch.cat((batch_samples_positive, near_perfect_trajectory), dim=1)


            # -----------------------------------------------------
            #                  Negative Samples
            # -----------------------------------------------------
            # set all zones in the vicinity of the neighbor pedestrians as negative samples

            # repeat x neighbors y agent zones times and reshape to (w=pred_length*x, y, nb_coords) tensor:
            # [N1, N2 ... Nx] -> [N11, N12 ... N1y, N21, N22 ... N2y, ...  ... ... ..., Nx1, Nx2 ... N2y]
            #                 -> [N11, N12 ... N1y;
            #                     N21, N22 ... N2y;
            #                     ...  ... ... ...;
            #                     Nw1, Nw2 ... Nwy] where N is [x,y]
            repeat_neighbors = np.repeat(neighbor_pedestrians.detach().cpu(), nb_zones, axis=1) \
                                 .reshape(self.pred_length*nb_neighbors, nb_zones, nb_coords)
            # tile agent zones w times and reshape to (w=pred_length*x, y, nb_coords) tensor:
            # [Z1, Z2 ... Zy] -> [Z11, Z12 ... Z1y, Z21, Z22 ... Z2y, ...  ... ... ..., Zx1, Zx2 ... Z2y].T
            #                 -> [Z11, Z12 ... Z1y;
            #                     Z21, Z22 ... Z2y;
            #                     ...  ... ... ...;
            #                     Zw1, Zw2 ... Zwy] where Z is [x,y]
            tile_agent_zones = np.tile(self.agent_zone, (self.pred_length*nb_neighbors, 1)) \
                                 .reshape(self.pred_length*nb_neighbors, nb_zones, nb_coords)
            # add a discomfort zone to every position and reshape to (pred_length, y*x, nb_coords)
            discomfort_zones = repeat_neighbors + tile_agent_zones
            discomfort_zones = discomfort_zones.reshape(self.pred_length, nb_zones*nb_neighbors, nb_coords)
            # append scene's negative samples to batch
            if batch_samples_negative is None:
                batch_samples_negative = discomfort_zones
            else:
                batch_samples_negative = torch.cat((batch_samples_negative, discomfort_zones), dim=1)

        #* DEBUG CODE SECTION ================================================================== *#
        '''
        # dimensions sanity check
        print("batch_pos shape", batch_samples_positive.shape) # nb batches primary pedestrians
        print("batch_neg shape", batch_samples_negative.shape) # on avg 4+ neighbor pedestrians
        # visualize effect on first batch, shows only prediction part (not observed)
        # uncomment end of gt_future = ... line at the beginning of function!
        traj_primary  = batch_scene[self.obs_length:, 0]
        traj_neighbor = batch_scene[self.obs_length:, 1:batch_split[1]]
        plot_scene(traj_primary, traj_neighbor, fname='scene_TESTCONTRASTIVE_original.png')
        traj_primary  = batch_samples_positive[:, 0]
        traj_neighbor = batch_samples_negative[:, 1:batch_split[1]*nb_zones]
        plot_scene(traj_primary, traj_neighbor, fname='scene_TESTCONTRASTIVE_modified.png')
        import pdb; pdb.set_trace()
        '''
        #* ===================================================================================== *#
        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------
        batch_samples_positive=batch_samples_positive.float()
        batch_samples_negative=batch_samples_negative.float()

        return batch_samples_positive, batch_samples_negative

class EventEncoder(nn.Module):
    '''
        Event encoder that maps an sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):

        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out

class SpatialEncoder(nn.Module):
    '''
        Spatial encoder that maps an sampled location to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state):
        return self.encoder(state)

class ProjHead(nn.Module):
    '''
        Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
            )

    def forward(self, feat):
        return self.head(feat)

def plot_scene(primary, neighbor, fname):
    '''
        Plot raw trajectories
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:, 0], primary[:, 1], 'k-')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')

    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
