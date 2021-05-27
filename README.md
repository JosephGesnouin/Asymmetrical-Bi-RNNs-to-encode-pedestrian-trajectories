# Asymmetrical Bi-RNNs to encode pedestrian trajectories
PyTorch implementation of Asymmetrical Bi-RNNs to encode pedestrian trajectories on trajnet++ dataset


TrajNet++ is a large scale interaction-centric trajectory forecasting benchmark comprising explicit agent-agent scenarios. Our code is built on top of the numerous baselines that are [available with Trajnet++](https://github.com/vita-epfl/trajnetplusplusbaselines).


## Training models

`python -m trajnetbaselines.rnns.trainer --arch [name of architecture] [USUAL ARGUMENTS]`

(the possible names are those of the files in the `trajnetbaselines/rnns/` folder, typically `u_lstm)


## Miscellaneous (not necessarily useful but I leave it just in case)

-- `ade_fde_plots` : to plot trajectories with ade and fde + plot the distribution of ade and fde + plot ade and fde as a function of distances and final angles
-- `average` : to average several existing predictions
-- `zip_for_submit`: everything is in the name

Beware of the "hard" paths I left in the first two scripts.


## Benchmarking Models:
