import itertools
import copy
import time
import numpy as np
import torch

import trajnetplusplustools

from .modules import Hidden2Normal, InputEmbedding

from .. import augmentation
from .utils import center_scene

NAN = float('nan')

def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM 
        """

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        ## Location
        scale = 4.0
        self.input_embedding = InputEmbedding(2, self.embedding_dim, scale)
        """
        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        #self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0
        """
        ## Pooling
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim 
        
        ## LSTMs
        self.encoder1 = torch.nn.LSTMCell(self.embedding_dim , self.hidden_dim)
        self.encoder2 = torch.nn.LSTMCell(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + pooling_dim, self.hidden_dim)

        #self.ln_e1 = torch.nn.LayerNorm(self.hidden_dim)
        #self.ln_e2 = torch.nn.LayerNorm(self.hidden_dim)
        #self.ln_d = torch.nn.LayerNorm(self.hidden_dim)

        # Predict the parameters of a multivariate normal:
        # mu_vel_x, mu_vel_y, sigma_vel_x, sigma_vel_y, rho
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_state, obs1, obs2, goals, batch_split, 
            hidden_only=False, h_inv=None, no_pool=False, ln=None):
        """Do one step of prediction: two inputs to one normal prediction.
        
        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_state : tuple (hidden_state, cell_state)
            Current hidden_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        
        Returns
        -------
        hidden_state : tuple (hidden_state, cell_state)
            Updated hidden_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position 
            with respect to the current position
        """
        
        num_tracks = len(obs2)
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        #print("N", num_tracks)
        #print("S", torch.sum(track_mask).item())
        #for (start, end) in zip(batch_split[:-1], batch_split[1:]):
        #    print(track_mask[start:end])

        ## Masked Hidden Cell State
        hidden_stacked = [torch.stack(list(itertools.compress(hidden_state[0], track_mask)), dim=0), 
                          torch.stack(list(itertools.compress(hidden_state[1], track_mask)), dim=0)]

        ## Mask current velocity & embed
        curr_velocity = obs2 - obs1
        curr_velocity = curr_velocity[track_mask]
        input_emb = self.input_embedding(curr_velocity)
            
        ## Mask & Pool per scene
        if self.pool is not None and not no_pool:
            hidden_states_to_pool = torch.stack(hidden_state[0]).clone() # detach?
            batch_pool = []
            ## Iterate over scenes
            for (start, end) in zip(batch_split[:-1], batch_split[1:]):
                ## Mask for the scene
                scene_track_mask = track_mask[start:end]
                ## Get observations and hidden-state for the scene
                prev_position = obs1[start:end][scene_track_mask]
                curr_position = obs2[start:end][scene_track_mask]
                curr_hidden_state = hidden_states_to_pool[start:end][scene_track_mask]

                ## Provide track_mask to the interaction encoders
                ## Everyone absent by default. Only those visible in current scene are present
                interaction_track_mask = torch.zeros(num_tracks, device=obs1.device).bool()
                interaction_track_mask[start:end] = track_mask[start:end]
                self.pool.track_mask = interaction_track_mask

                ## Pool
                pool_sample = self.pool(curr_hidden_state, prev_position, curr_position)
                batch_pool.append(pool_sample)

            pooled = torch.cat(batch_pool)
            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_stacked[0] += pooled

        if h_inv is not None:
            h_inv = torch.stack(list(itertools.compress(h_inv, track_mask)), dim=0)
            input_emb = torch.cat([input_emb, h_inv], dim=1)

        # LSTM step
        hidden_stacked = lstm(input_emb, hidden_stacked)

        if ln is not None:
            hidden_stacked = ln(hidden_stacked)

        mask_index = list(itertools.compress(range(len(track_mask)), track_mask))

        if not hidden_only:
            normal_masked = self.hidden2normal(hidden_stacked[0])
        # unmask [Update hidden-states and next velocities of pedestrians]
            normal = torch.full((track_mask.size(0), 5), NAN, device=obs1.device)  
            for i, h, c, n in zip(mask_index,
                              hidden_stacked[0],
                              hidden_stacked[1],
                              normal_masked):
                hidden_state[0][i] = h
                hidden_state[1][i] = c
                normal[i] = n
            return hidden_state, normal
        
        else:
            for i, h, c in zip(mask_index,
                              hidden_stacked[0],
                              hidden_stacked[1]):
                hidden_state[0][i] = h
                hidden_state[1][i] = c
            return hidden_state
        

    def forward(self, observed, goals, batch_split, prediction_truth=None, n_predict=None):
        """Forecast the entire sequence 
        
        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2]
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene        
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """
        #print(observed.shape)
        #print(None if prediction_truth is None else prediction_truth.shape)
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict-1)] #!!!

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(1)
        hidden_state = ([torch.zeros(self.hidden_dim, device=observed.device) 
                            for _ in range(num_tracks)],
                        [torch.zeros(self.hidden_dim, device=observed.device) 
                            for _ in range(num_tracks)])
        ## Reset LSTMs of Interaction Encoders.
        if self.pool is not None:
            self.pool.reset(num_tracks, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        h_invs=[]
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step
            hidden_state = self.step(self.encoder1, hidden_state, obs1, obs2, goals, batch_split, 
                            hidden_only=True, no_pool=True)#, ln=self.ln_e1)
            h_invs.append(hidden_state[0])

        hidden_state = ([torch.zeros(self.hidden_dim, device=observed.device) 
                            for _ in range(num_tracks)],
                        [torch.zeros(self.hidden_dim, device=observed.device) 
                            for _ in range(num_tracks)])

        inv_observed = torch.flip(observed, (0,))
        for obs1, obs2, h_inv in zip(inv_observed[1:], inv_observed[:-1], h_invs[::-1]):
            ##LSTM Step
            hidden_state = self.step(self.encoder2, hidden_state, obs1, obs2, goals, batch_split, 
                            hidden_only=True, h_inv=h_inv, no_pool=True)#, ln=self.ln_e2)
            # concat predictions
            #!!!normals.append(normal)
            #!!!positions.append(obs2 + normal[:, :2])  # no sampling, just mean
        #print(obs1.shape)
        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-2:], prediction_truth)
        )))

        # decoder, predictions
        i=0
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            elif i>=2:
                for primary_id in batch_split[:-1]:
                    obs1[primary_id] = positions[-2][primary_id].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            elif i>=1:
                for primary_id in batch_split[:-1]:
                    obs2[primary_id] = positions[-1][primary_id].detach()  # DETACH!!!
            hidden_state, normal = self.step(self.decoder, hidden_state, obs1, obs2, goals, batch_split)#, ln=self.ln_d)
            i+=1
            # concat predictions
            normals.append(normal)
            positions.append(obs2 + normal[:, :2])  # no sampling, just mean
        #print(obs1.shape)

        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)
        #print(rel_pred_scene.shape)
        return rel_pred_scene, pred_scene

class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f)


    def __call__(self, paths, scene_goal, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):
        self.model.eval()
        # self.model.train()
        with torch.no_grad():
            xy = trajnetplusplustools.Reader.paths_to_xy(paths)
            # xy = augmentation.add_noise(xy, thresh=args.thresh, ped=args.ped_type)
            batch_split = [0, xy.shape[1]]

            if args.normalize_scene:
                xy, rotation, center, scene_goal = center_scene(xy, obs_length, goals=scene_goal)
            
            xy = torch.Tensor(xy)  #.to(self.device)
            scene_goal = torch.Tensor(scene_goal) #.to(device)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            for num_p in range(modes):
                # _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, xy[obs_length:-1].clone())
                _, output_scenes = self.model(xy[start_length:obs_length], scene_goal, batch_split, n_predict=n_predict)
                output_scenes = output_scenes.numpy()
                if args.normalize_scene:
                    output_scenes = augmentation.inverse_scene(output_scenes, rotation, center)
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                ## Dictionary of predictions. Each key corresponds to one mode
                multimodal_outputs[num_p] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs
