import torch
from torch import nn
from torch.nn import functional as F
from itertools import combinations

from utils.helpers import StateEncoder


class BaseDiscardNet(nn.Module):
    """ Base neural network for training discard policies. """

    INPUT_SIZE = 105
    OUTPUT_SIZE = 15
    DISCARD_COMBO_ORDER = tuple(enumerate(combinations(range(6), 2)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self) -> None:
        """ Create a new BaseDiscardNet instance. """

        super().__init__()

        self.net = None


    def load_weights(self, file_name: str) -> None:
        """
        Load the weights from a pretrained network (at trained_nets/discard_nets/file_name.pt).

        ------

        Arguments:
            file_name: The file name to load from.
        """

        self.net.load_state_dict(torch.load(f'trained_nets/discard_nets/{file_name}.pt', map_location = self.device))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Send an input though the net and get an output. """

        return self.net(x)


    def get_distribution_policy(self, player_score: int, opponent_score: int, is_dealer: bool,
                                player_hand: list[str]) -> list[tuple[str, str, torch.Tensor]]:
        """
        Get the processed output of the network for a given input.

        ------

        Arguments:
            player_score: The neural network player's score.
            opponent_score: The opponent's score.
            is_dealer: Whether the neural network player is the dealer.
            player_hand: The neural network player's cards in hand.

        ------

        Returns:
            A list of all 15 discard combinations along with their log-softmax values.
        """

        encoded_state = StateEncoder.encode_state_for_discard_phase(
            player_score, opponent_score, is_dealer, player_hand
        )

        outputs = self.net(torch.tensor(encoded_state, dtype = torch.float32, device = self.device))
        probs = F.log_softmax(outputs, dim = -1)

        discard_combos = []
        for idx, (card1, card2) in self.DISCARD_COMBO_ORDER:
            discard_combos.append((player_hand[card1], player_hand[card2], probs[idx]))

        return discard_combos


    @staticmethod
    def get_combo_confidence(discard_combos: list[tuple[str, str, torch.Tensor]],
                             card1: str, card2: str) -> torch.Tensor | None:
        """
        Get the confidence for a specific discard combination from the given network output.

        ------

        Arguments:
            discard_combos: A processed output from the network to search in.
            card1: First card of the discard combination.
            card2: Second card of the discard combination.

        ------

        Returns:
            The confidence score for the specified discard combination or None if not found.
        """

        for combo in discard_combos:
            if card1 in combo and card2 in combo:
                return combo[2]


    def get_discard_action(self, player_score: int, opponent_score: int, is_dealer: bool,
                           player_hand: list[str]) -> tuple[str, str, torch.Tensor]:
        """
        Choose which cards to discard based on the given state.

        ------

        Arguments:
            player_score: The neural network player's score.
            opponent_score: The opponent's score.
            is_dealer: Whether the neural network player is the dealer.
            player_hand: The neural network player's cards in hand.

        ------

        Returns:
            The combination of cards chosen to be discarded along with their confidence score.
        """

        distribution = self.get_distribution_policy(player_score, opponent_score, is_dealer, player_hand)
        best_combo = max(distribution, key = lambda x: x[2].item())
        return best_combo[0], best_combo[1], best_combo[2]


__all__ = ['BaseDiscardNet']
