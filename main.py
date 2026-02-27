from utils.assets import Display
from utils.players import RandomPlayer, UserPlayer, AnalyticalDiscardPlayer, DNPRPlayer
from utils.game import Game

from utils.neural_nets import DiscardNetV1, DiscardTrainer


net = DiscardNetV1()
DiscardTrainer.train(
    net,
    play_style = 'recommended',
    lr = 1e-3,
    wd = 1e-4,
    epochs = 1,
    alpha = 1,
    alpha_step = 10,
    alpha_decay = 0.1,
    early_stop = False
)
DiscardTrainer.save(
    net, f'discard_net_v1_test',
    comment = 'Test Test :3',
    logs = True
)


if __name__ == '__main__':
    input('... preventing program from continuing by waiting for input ...\n'
          '... full-screen the terminal before continuing ...')

    game = Game(RandomPlayer(),
                AnalyticalDiscardPlayer(),
                wait_after_move = 'input',
                wait_after_info = False,
                show_opponents_hand = False)

    game.play()
