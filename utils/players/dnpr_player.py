import random

from utils.players import BasePlayer
from utils.neural_nets import BaseDiscardNet

class DNPRPlayer(BasePlayer):
    """ Player agent that uses a nural network during the discard phase and plays randomly during the pegging phase. """

    def __init__(self, discard_net: BaseDiscardNet) -> None:
        """
        Create a new DNPRPlayer instance.

        ------

        Arguments:
            discard_net: A pretrained discard network.
        """
        super().__init__()
        self.discard_net = discard_net
        self.discard_net.net.eval()


    def discard_cards(self, state: dict[str, ...]) -> list[str]:

        opponent = state['player1'] if self == state['player2'] else state['player2']
        is_dealer = state['dealer'] == self

        card1, card2, _ = self.discard_net.get_discard_action(self.points, opponent.points, is_dealer, self.cards)

        self.cards.remove(card1)
        self.cards.remove(card2)

        return [card1, card2]


    def play_card(self, state: dict[str, ...]) -> str:

        card = random.choice(self.get_valid_moves(state))

        if card != "GO":
            self.cards.remove(card)

        return card



__all__ = ['DNPRPlayer']
