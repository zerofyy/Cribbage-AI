import itertools

from utils.helpers import CardDeck, Scoring


class DiscardEvaluator:
    """ Helper for discarding cards based on statistical probability. """

    _precomputed_scores: dict[tuple[str, ...], int] = {}
    _shift_key_by: dict[str, int] = {rank : 1 << (i * 3) for i, rank in enumerate(CardDeck.CARD_RANKS)}


    @classmethod
    def _get_cards_key(cls, cards: list[str]) -> tuple[str, ...]:
        """
        Get a key for the precomputed scores dictionary based on the given cards.

        ------

        Arguments:
            cards: A list of cards in rank-suit format.

        ------

        Returns:
            The dictionary key.
        """

        key = 0
        for card in cards:
            key += cls._shift_key_by[card[0]]

        return key


    @classmethod
    def _precompute_scores(cls) -> None:
        """ Precompute scores for all possible 5-card hand combinations, ignoring suits. """

        if cls._precomputed_scores:
            return

        for combo in itertools.combinations_with_replacement(CardDeck.CARD_RANKS, 5):
            if any(combo.count(card) > 4 for card in set(combo)):
                continue

            cards = list(combo)
            cls._precomputed_scores[cls._get_cards_key(cards)] = \
                Scoring.score_15(cards, 'hand')[0] + \
                Scoring.score_run(cards, 'hand')[0] + \
                Scoring.score_pair(cards, 'hand')[0]


    @classmethod
    def score_hand(cls, cards: list[str], starter_card: str) -> int:
        """
        Calculate the score of a given hand and starter card using precomputed hand scores.

        ------

        Arguments:
             cards: List of 4 cards in rank-suit format.
             starter_card: The starter card in rank-suit format.

        ------

        Returns:
            The hand's score.
        """

        if not cls._precomputed_scores:
            cls._precompute_scores()

        score_key = cls._get_cards_key(cards + [starter_card])
        score = cls._precomputed_scores[score_key]

        # Flush
        if cards[0][1] == cards[1][1] == cards[2][1] == cards[3][1]:
            score += 4
            if starter_card[1] == cards[0][1]:
                score += 1

        # His nobs
        for card in cards:
            if card[0] == 'J' and starter_card[1] == card[1]:
                score += 2
                break

        return score


    @classmethod
    def score_crib(cls, cards: list[str], starter_card: str) -> int:
        """
        Calculate the score of a given crib and starter card using precomputed hand scores.

        ------

        Arguments:
             cards: List of 4 cards in rank-suit format.
             starter_card: The starter card in rank-suit format.

        ------

        Returns:
            The crib's score.
        """

        if not cls._precomputed_scores:
            cls._precompute_scores()

        score_key = cls._get_cards_key(cards + [starter_card])
        score = cls._precomputed_scores[score_key]

        # Flush
        if cards[0][1] == cards[1][1] == cards[2][1] == cards[3][1] == starter_card[1]:
            score += 5

        # His heels
        if starter_card[0] == 'J':
            score += 2

        return score


    @classmethod
    def get_discard_stats(cls, hand: list[str], is_dealer: bool) -> dict[str, list]:
        """
        Get discard suggestions for a given hand.

        ------

        Arguments:
             hand: List of 6 cards in rank-suit format.
             is_dealer: Whether the hand belongs to the dealer.

        ------

        Returns:
            A dictionary of different discard suggestions, sorted from best to worst (by their criteria).
        """

        discard_combos = {}
        deck = CardDeck(shuffle = False)
        deck = [card for card in deck.cards if card not in hand]
        for my_discard in itertools.combinations(hand, 2):
            remaining_hand = [card for card in hand if card not in my_discard]

            all_disc_scores = []
            sum_disc_scores = 0
            sum_hand_scores = 0
            max_score = float('-inf')
            min_score = float('inf')
            for i, starter_card in enumerate(deck):

                hand_score = cls.score_hand(remaining_hand, starter_card)
                sum_hand_scores += hand_score

                remaining_deck = deck[:i] + deck[i + 1:]
                for opp_discard in itertools.combinations(remaining_deck, 2):
                    crib_cards = list(my_discard + opp_discard)
                    crib_score = cls.score_crib(crib_cards, starter_card)

                    disc_score = hand_score + crib_score if is_dealer else hand_score - crib_score
                    sum_disc_scores += disc_score
                    if disc_score > max_score:
                        max_score = disc_score
                    if disc_score < min_score:
                        min_score = disc_score
                    all_disc_scores.append(disc_score)

            avg_score = sum_disc_scores / len(all_disc_scores)
            hand_score = sum_hand_scores / len(deck)
            high_score = sorted(all_disc_scores)[int(len(all_disc_scores) * 0.95)]

            discard_combos[my_discard] = {
                'avg' : avg_score, 'min' : min_score, 'max' : max_score, 'hand' : hand_score, 'high' : high_score
            }

        # Recommended: Card pairs sorted by the highest average when discarded.
        #              If there are multiple combinations with the same average, take the one with the
        #              highest hand score.
        recommended = [(cards, scores['avg']) for cards, scores in sorted(
            discard_combos.items(), key = lambda item: (item[1]['avg'], item[1]['hand']), reverse = True)]

        # For the following plays, if there are multiple combinations with the same value (min, max, ...),
        # take the one with the highest average.

        # Sure Bet: Card pairs sorted by the highest minimum when discarded.
        sure_bet = [(cards, scores['min']) for cards, scores in sorted(
            discard_combos.items(), key = lambda item: (item[1]['min'], item[1]['avg']), reverse = True)]

        # Risky Bet: Card pairs sorted by the highest maximum when discarded.
        risky_bet = [(cards, scores['max']) for cards, scores in sorted(
            discard_combos.items(), key = lambda item: (item[1]['max'], item[1]['avg']), reverse = True)]

        # Hail Mary: Card pairs sorted by their best shot at a high score when discarded.
        #            Where the high score is X points or more, 5% of the time.
        hail_mary = [(cards, scores['high']) for cards, scores in sorted(
            discard_combos.items(), key = lambda item: (item[1]['high'], item[1]['avg']), reverse = True)]

        # Aggressive: Card pairs sorted by the highest average hand score when discarded.
        aggressive = [(cards, scores['hand']) for cards, scores in sorted(
            discard_combos.items(), key = lambda item: (item[1]['hand'], item[1]['avg']), reverse = True)]

        return {
            'recommended' : recommended,
            'sure_bet' : sure_bet,
            'risky_bet' : risky_bet,
            'hail_mary' : hail_mary,
            'aggressive' : aggressive
        }


__all__ = ['DiscardEvaluator']
