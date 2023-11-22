from trainers.trainer import Trainer
import os

if __name__ == "__main__":
    setup = {'agents': [(3, 'DDPG'),],
             'discrete_action': False,
             'train_env': 'simple_adversary',
             'eval_env': 'simple_adversary'}

    # get root dir
    root_dir = os.path.dirname(os.path.realpath(__file__))

    trainer = Trainer(setup, root_dir)
    trainer.train()