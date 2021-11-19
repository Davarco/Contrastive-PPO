# Contrastive Pre-Training for Reinforcement Learning

We tackle sample-efficiency and generalization in RL using contrastive pretraining. We test our results on the FruitBot game from OpenAI's ProcGen environments, which are specifically designed to evaluate these metrics. 

Our model is split into two parts. First, there is a game state encoder which turns an image into a feature vector. The actor and critic network are just a linear layer on top of these features. When training the contrastive loss, we reuse the encoder and use augmented images from FruitBot. More details can be found in our [final report](http://github.com/Davarco/Contrastive-PPO/cs182_final_report.pdf). With regards to data augmentations, we find that crops + color distortion works best. 

#### Project Setup
First, clone the repository. Then, create a new `conda` environment and run the following command.

`pip install -r requirements.txt`

#### Experiments
All the commands to run experiments are in `experiments.sh`. The models are saved to the `models/` directory, and tensorboard logs to the `logs/` directory.

