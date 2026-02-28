import random

import torch
from torch import optim

from utils.neural_nets import BaseDiscardNet
from utils.helpers import DiscardEvaluator, CardDeck

from multiprocessing import Pool, cpu_count


def _get_batch_data(args: dict[str, ...]) -> dict[str, ...]:
    """
    Generate input data for a single training batch.

    ------

    Arguments:
        args: A dictionary containing training info.

    ------

    Returns:
        A dictionary with the generated training data.
    """

    play_style = args['play_style']
    alpha = args['alpha']

    deck = CardDeck(shuffle=True)
    hand_cards, crib_cards, starter_card = deck.deal_cards(6), deck.deal_cards(2), deck.deal_cards(1)[0]
    is_dealer = random.choice([True, False])
    score1, score2 = random.randint(0, 121), random.randint(0, 121)

    best_cards = None
    if alpha > 0:
        ranked_pairs = DiscardEvaluator.get_discard_stats(hand_cards, is_dealer)[play_style]
        best_cards = ranked_pairs[0][0]

    return {
        'score1': score1,
        'score2': score2,
        'is_dealer': is_dealer,
        'hand_cards': hand_cards,
        'crib_cards': crib_cards,
        'starter_card': starter_card,
        'best_cards': best_cards
    }



class DiscardTrainer:
    """ Neural network trainer for the discarding phase. """

    BEST_HAND_SCORE = 29  # four 5s and a Jack with the same suit as the starter card
    BEST_CRIB_SCORE = 29  # same as hand
    BEST_OUTCOME = 53     # best hand + 6644 in crib
    WORST_OUTCOME = 0 - BEST_CRIB_SCORE  # nothing scored in hand
    _training_logs: str = ''


    @classmethod
    def _log(cls, text: str) -> None:
        """
        Print and expand the training logs.

        ------

        Arguments:
             text: The log text.
        """

        cls._training_logs += f'{text}\n'
        print(text)


    @classmethod
    def train(cls, discard_network: BaseDiscardNet, lr: float, wd: float, epochs: int, batch_size: int = 32,
              early_stop: bool = True, play_style: str = None, alpha: float = 0.0, alpha_decay: float = 0.95,
              alpha_step: int = 10, num_workers: int = 1) -> None:
        """
        Train the given discard neural network using reinforced supervised or unsupervised learning.

        Unsupervised is used when alpha = 0. While supervised is used when play_style != None and alpha > 0.

        ------

        Available play styles:
            - recommended: Reward based on statistical highest average score.
            - sure_bet: Reward based on statistical highest minimum score.
            - risky_bet: Reward based on statistical highest maximum score.
            - hail_mary: Reward based on statistical best chance at a high score.
            - aggressive: Reward based on statistical highest hand score.

        ------

        Arguments:
            discard_network: The discard network to be trained.
            lr: The learning rate.
            wd: The weight decay.
            epochs: The number of epochs.
            batch_size: The number of batches per epoch.
            early_stop: Whether to stop training early if results are satisfactory.
            play_style: The play style the net should be trained in.
            alpha: How reliant the network is on the given play-style initially.
            alpha_decay: By how much to reduce the network's dependency of the given play-style.
            alpha_step: How often to decay alpha in epochs.
            num_workers: How many cores to use for training.

        ------

        Returns:
            The trained discard network.
        """

        cls._training_logs = ''

        cls._log(
            f'{discard_network.__class__.__name__} Structure Details:\n'
            f'{discard_network.net}\n\n'
        )

        cls._log(
            f'{discard_network.__class__.__name__} Training Details:\n'
            f'* Learning Rate: {lr}\n'
            f'* Weight Decay: {wd}\n'
            f'* Epochs: {epochs}\n'
            f'* Batch Size: {batch_size}\n'
            f'\n'
            f'* Early Stopping: {early_stop}\n'
            f'\n'
            f'* Supervised: {alpha > 0}\n'
            f'* Alpha: {alpha}\n'
            f'* Alpha Decay: {alpha_decay}\n'
            f'* Alpha Step: {alpha_step}\n\n'
        )

        epoch_spaces = len(str(epochs))

        net = discard_network.net
        optimizer = optim.AdamW(net.parameters(), lr = lr, weight_decay = wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-5)

        num_workers = min(cpu_count(), num_workers)
        if batch_size % num_workers != 0:
            cls._log(f'Batch size {batch_size} cant be evenly distributed to cores.')
            batch_size = (batch_size // num_workers) * num_workers
            cls._log(f'Changed batch size to {batch_size}.')

        cls._log(f'Training with {discard_network.device}...')
        net.train()

        with Pool(processes = num_workers) as pool:
            for epoch in range(1, epochs + 1):
                total_loss, total_reward, total_reward_norm = 0, 0, 0

                batch_args = [{'play_style': play_style, 'alpha': alpha} for _ in range(batch_size)]
                batch_results = pool.map(_get_batch_data, batch_args)

                for batch in batch_results:
                    hand_cards = batch['hand_cards']
                    crib_cards = batch['crib_cards']
                    is_dealer = batch['is_dealer']
                    starter_card = batch['starter_card']

                    distribution = discard_network.get_distribution_policy(batch['score1'], batch['score2'],
                                                                           is_dealer, hand_cards)
                    best_combo = max(distribution, key=lambda x: x[2].item())
                    card1, card2, confidence = best_combo[0], best_combo[1], best_combo[2]

                    imitation_loss = 0
                    if alpha > 0:
                        best_pair = batch['best_cards']
                        imitation_loss = -discard_network.get_combo_confidence(
                            distribution, card1=best_pair[0], card2=best_pair[1]
                        )

                    hand_cards.remove(card1)
                    hand_cards.remove(card2)
                    crib_cards.extend([card1, card2])
                    hand_score = DiscardEvaluator.score_hand(hand_cards, starter_card)
                    crib_score = DiscardEvaluator.score_crib(crib_cards, starter_card)

                    reward = hand_score + crib_score if is_dealer else hand_score - crib_score
                    normalized_reward = (reward - cls.WORST_OUTCOME) / (cls.BEST_OUTCOME - cls.WORST_OUTCOME)
                    rl_loss = -(1 / normalized_reward) * confidence

                    loss = alpha * imitation_loss + (1 - alpha) * rl_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_reward += reward
                    total_reward_norm += normalized_reward

                scheduler.step()

                avg_loss = total_loss / batch_size
                avg_reward = total_reward / batch_size
                avg_reward_norm = total_reward_norm / batch_size

                if epoch % alpha_step == 0 and alpha > 0:
                    alpha = max(alpha - alpha_decay, 0)

                cls._log(f'* Epoch: {epoch:<{epoch_spaces}}  |'
                         f'  L: {avg_loss:<4.8f}'
                         f'  R: {avg_reward:<4.8f}'
                         f'  RN: {avg_reward_norm:<4.8f}'
                         f'  A: {alpha}')

        cls._log('Training finished.')


    @classmethod
    def save(cls, discard_network: BaseDiscardNet, file_name: str, comment: str, logs: bool = True) -> None:
        """
        Save the trained discard network at /trained_nets/discard_nets/file_name.pt.

        ------

        Arguments:
            discard_network: The trained discard network.
            file_name: The file name
            comment: A comment to be saved along with the weights file.
            logs: Whether to include the training logs.
        """

        torch.save(discard_network.net.state_dict(), f'trained_nets/discard_nets/{file_name}.pt')

        if logs:
            comment += f'\n\n{cls._training_logs}'

        with open(f'trained_nets/discard_nets/{file_name}_comment.txt', 'w') as file:
            file.write(comment)


__all__ = ['DiscardTrainer']
