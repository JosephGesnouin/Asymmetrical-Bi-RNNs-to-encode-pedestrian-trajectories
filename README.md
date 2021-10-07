# Asymmetrical Bi-RNNs to encode pedestrian trajectories
PyTorch implementation of Asymmetrical Bi-RNNs to encode pedestrian trajectories on the Trajnet++ dataset

2nd place solution of the Trajnet++ Challenge during the Long-term Human Motion Prediction Workshop, IEEE International Conference on Robotics and Automation (ICRA 2021)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/asymmetrical-bi-rnn-for-pedestrian-trajectory/trajectory-prediction-on-trajnet)](https://paperswithcode.com/sota/trajectory-prediction-on-trajnet?p=asymmetrical-bi-rnn-for-pedestrian-trajectory)

3rd place solution of the Trajnet++ Challenge during the Multi-Agent Interaction and Relational Reasoning Workshop, IEEE International Conference on Computer Vision (ICCV 2021)


## Idea developped in the paper:
Contrary to many previous studies which proposed new interactions modules but did not deepen the importance of a robust sequence encoder, our work solely
rely on proposing a new sequence encoder that could be easily applicable on all models that use the encoder-decoder pipeline for pedestrian trajectory forecasting while taking advantage of the research on interactions.

We propose an Asymmetrical Bi-RNNs architecture to replace regular LSTMs or Bi-LSTMs as a motion-encoding baseline for pedestrian trajectories forecasting:

![Asymmetrical Bi-RNNs architecture](https://github.com/JosephGesnouin/Asymmetrical-Bi-RNNs-to-encode-pedestrian-trajectories/blob/main/u-rnn.png)

-- An aspect of Bi-RNNs that could be undesirable is the architecture's symmetry in both time directions. Bi-RNNs are often used in natural language processing, where the order of the words is almost exclusively determined by grammatical rules and not by temporal sequentiality. However, in trajectory prediction, the data has a preferred direction in time: the forward direction. 

-- Another potential drawback of Bi-RNNs is that their output is simply the concatenation of two naive readings of the input in both directions. In consequence, Bi-RNNs never actually read an input by knowing what happens in the future. Conversely the idea behind our approach, is to first do a backward pass, and then use during the forward pass information about the future. By using an asymmetrical Bi-RNN to encode pedestrian trajectories, we accumulate information while knowing which part of the information will be useful in the future as it should be relevant to do so if the forward direction is the preferred direction of the data.

## Data set


TrajNet++ is a large scale interaction-centric trajectory forecasting benchmark comprising explicit agent-agent scenarios. Our code is built on top of the numerous baselines that are [available with Trajnet++](https://github.com/vita-epfl/trajnetplusplusbaselines).

If you want to replicate our results, follow the [guidelines from the Trajnet++ benchmark hosts](https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/) to ensure you are good to go on the Trajnet++ dataset, thereafter fork our repository with respect to its architecture (/rnns/) and follow the guidelines for training our models.



## Training models

`python -m trajnetbaselines.rnns.trainer --arch [name of architecture] [USUAL ARGUMENTS]`

The possible names are those of the files in the `trajnetbaselines/rnns/` folder, typically `u_lstm`).
All the architectures listed in the paper are available: Bi-RNNs, RNNs, U-RNNs (Asymmetrical ours), Reversed U-RNNs.

We also present a version of the Social NCE contrastive loss that is **NOT** the official implementation of the paper


## Miscellaneous (not necessarily useful but just in case)

-- `ade_fde_plots` : to plot trajectories with ade and fde + plot the distribution of ade and fde + plot ade and fde as a function of distances and final angles
-- `average` : to average several existing predictions


Beware of the "hard" paths we left in the two scripts.


## Benchmarking Models:
The challenge is hosted on [AIcrowds](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge), please follow the [guidelines](https://thedebugger811.github.io/posts/2021/04/milestone_1/) to create a submission

## Citation

If you find this code useful in your research, please consider citing:

```bash
@inproceedings{rozenberg:hal-03244860,
  TITLE = {{Asymmetrical Bi-RNNs (U-RNNs), 2nd place solution at the Trajnet++ Challenge for pedestrian trajectory forecasting}},
  AUTHOR = {Rozenberg, Rapha{\"e}l and Gesnouin, Joseph and Moutarde, Fabien},
  URL = {https://hal-mines-paristech.archives-ouvertes.fr/hal-03244860},
  BOOKTITLE = {{Workshop on Long-term Human Motion Prediction, 2021 IEEE International Conference on Robotics and Automation (ICRA)}},
  ADDRESS = {Xi'an, China},
  YEAR = {2021},
  MONTH = May,
  HAL_ID = {hal-03244860},
  HAL_VERSION = {v1},
}

@ARTICLE{2021arXiv210604419R,
       author = {{Rozenberg}, Rapha{\"e}l and {Gesnouin}, Joseph and {Moutarde}, Fabien},
        title = "{Asymmetrical Bi-RNN for pedestrian trajectory encoding}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Artificial Intelligence},
         year = 2021,
        month = jun,
          eid = {arXiv:2106.04419},
        pages = {arXiv:2106.04419},
archivePrefix = {arXiv},
       eprint = {2106.04419},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021arXiv210604419R},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}

  ```

