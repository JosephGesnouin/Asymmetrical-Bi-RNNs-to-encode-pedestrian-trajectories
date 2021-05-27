# Asymmetrical Bi-RNNs to encode pedestrian trajectories
PyTorch implementation of Asymmetrical Bi-RNNs to encode pedestrian trajectories on the Trajnet++ dataset


## Idea developped in the paper:
Contrary to many previous studies which proposed new interactions modules but did not deepen the importance of a robust sequence encoder, our work solely
rely on proposing a new sequence encoder that could be easily applicable on all models that use the encoder-decoder pipeline for pedestrian trajectory forecasting while taking advantage of the research on interactions.

We propose an Asymmetrical Bi-RNNs architecture to replace regular LSTMs or Bi-LSTMs as a motion-encodoing baseline for pedestrian trajectories forecasting:

![Asymmetrical Bi-RNNs architecture](https://github.com/JosephGesnouin/Asymmetrical-Bi-RNNs-to-encode-pedestrian-trajectories/blob/main/u-rnn.png)

An aspect of Bi-RNNs that could be undesirable is the architecture's symmetry in both time directions. Bi-RNNs are often used in natural language processing, where the order of the words is almost exclusively determined by grammatical rules and not by temporal sequentiality. However, in trajectory prediction, the data has a preferred direction in time: the forward direction. Another potential drawback of Bi-RNNs is that their output is simply the concatenation of two naive readings of the input in both directions. In consequence, Bi-RNNs never actually read an input by knowing what happens in the future. Conversely the idea behind our approach, is to first do a backward pass, and then use during the forward pass information about the future. By using an asymmetrical Bi-RNN to encode pedestrian trajectories, we accumulate information while knowing which part of the information will be useful in the future as it should be relevant to do so if the forward direction is the preferred direction of the data.

## Data set


TrajNet++ is a large scale interaction-centric trajectory forecasting benchmark comprising explicit agent-agent scenarios. Our code is built on top of the numerous baselines that are [available with Trajnet++](https://github.com/vita-epfl/trajnetplusplusbaselines).


## Training models

`python -m trajnetbaselines.rnns.trainer --arch [name of architecture] [USUAL ARGUMENTS]`

the possible names are those of the files in the `trajnetbaselines/rnns/` folder, typically `u_lstm`)


## Miscellaneous (not necessarily useful but just in case)

-- `ade_fde_plots` : to plot trajectories with ade and fde + plot the distribution of ade and fde + plot ade and fde as a function of distances and final angles
-- `average` : to average several existing predictions
-- `zip_for_submit`: everything is in the name

Beware of the "hard" paths I left in the first two scripts.


## Benchmarking Models:


## Citation

If you find this code useful in your research, then please cite:
