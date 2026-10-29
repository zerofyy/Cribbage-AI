<h1 align="center">
  Cribbage AI
</h1>

<p align="center">
  Cribbage card game playable in the terminal with a framework for training neural network agents using supervised reinforcement learning, including a statistical discard analyzer.
  <br>
  This project was developed in collaboration with <a href="https://github.com/ilina-d">zhap</a> 🙂
</p>

<br>

<p align="center">
  <a href="#how-to-play">How to Play</a> •
  <a href="#usage">Usage</a> •
  <a href="#implemented-players">Implemented Players</a> •
  <a href="#neural-networks">Neural Networks</a> •
  <a href="#results">Results</a> •
  <a href="#fun-facts">Fun Facts</a>
</p>

<p align="center"><img src="https://github.com/user-attachments/assets/8d4475d2-eec9-4789-ab19-cc940b04fe1e"></p>

<br>

## How to Play
Cribbage is a two-player card game played with a standard 52-card deck (no jokers). The game is made up of an arbitrary number of rounds, in which players alternate the role of **dealer** and score points by playing their cards. The first player to reach **121 points** wins.

At the start of each round the dealer shuffles and deals **6 cards** to each player. Every round then plays out in three phases:

### Phase 1: Discarding
Both players choose **2 cards** to remove from their hand. The 4 discarded cards form **the dealer's crib**, which stays face down until the Show phase and counts for the dealer.

### Phase 2: Pegging
A **starter card** is drawn from the deck and turned face up. Players then take turns playing one card at a time (starting with the non-dealer), keeping a running sum that must not exceed **31** (face cards are worth 10, aces are worth 1). If a player can't play without going over 31, they say **GO** and the opponent continues. When neither player can play, the count resets and play continues until all 8 cards are played.

Points are scored for the following *tricks*:

| Trick       | Points                                                       |
|-------------|--------------------------------------------------------------|
| Fifteen     | **2** for making the running sum exactly 15                  |
| Thirty-one  | **2** for making the running sum exactly 31                  |
| Pairs       | **2 / 6 / 12** for a pair / three / four of a kind in a row  |
| Runs        | **1 point per card** for a run of 3 or more consecutive ranks (in any order) |
| Go          | **1** for the opponent when a player is forced to say GO     |
| Last card   | **1** for playing the last card                              |
| His Heels   | **2** for the dealer if the starter card is a Jack           |

### Phase 3: Show
Each player scores their remaining 4-card hand together with the starter card as a whole, followed by the dealer scoring their crib. Points are awarded on the same principle as in pegging, with a few changes:

| Trick       | Points                                                                                   |
|-------------|------------------------------------------------------------------------------------------|
| Fifteen     | **2** for each combination of cards that sums to 15                                      |
| Pairs       | **2** for each pair                                                                      |
| Runs        | **1 point per card** for each run of 3+ consecutive ranks. Double, triple and quadruple runs (duplicate cards) simply count as 2, 3 or 4 separate runs |
| Flush       | **4** if all 4 hand cards share a suit, **+1** if the starter card has that suit too     |
| His Nobs    | **2** for holding a Jack with the same suit as the starter card                          |

After both hands are scored, the dealer flips **the dealer's crib** and scores it in the same way, with two differences:
* **Flush:** there is no 4-card flush. A flush only counts if all 4 crib cards **and** the starter share a suit (**5** points).
* **His Nobs:** is not scored in the crib.

After the Show phase the dealer role passes to the opponent and a new round begins.

<br>

## Usage
### Installation
```bash
# Clone the repository
git clone https://github.com/zerofyy/Cribbage-AI.git
cd Cribbage-AI

# Install dependencies
pip install -r requirements.txt

# Run
python main.py
```

### Running the Game
```py
if __name__ == '__main__':

    input('... preventing program from starting by waiting for input ...\n'
          '... full-screen the terminal before continuing ...')

    game = Game(
        player1 = UserPlayer(),
        player2 = DAPGPlayer(),
        first_dealer = None,
        wait_after_move = 500,
        wait_after_info = True,
        visuals = True
    )

    game.play()

    input('... preventing program from continuing by waiting for input ...')
```

### Using the Discard Evaluator
```py
    net = DNT_ShallowSeluWide()
    net.load_weights('example_weights')

    game = Game(
        player1 = UserPlayer(),
        player2 = DNPRPlayer(net),
        first_dealer = None,
        wait_after_move = 500,
        wait_after_info = True,
        show_opponents_hand = False,
        visuals = True,
        measure_statistics = False
    )

    game.play()
```

The `DiscardEvaluator` is a reconstruction of [C. Liam Brown's](https://cliambrown.com/cribbage/) statistical discard analysis algorithm. For a given 6-card hand and dealer status, it examines every possible starter card and every possible crib completion to calculate the expected points of each of the 15 discard options. It supports several play styles:

| Style           | Sorted by                                                  |
|-----------------|------------------------------------------------------------|
| `recommended`   | Highest average points (ties broken by hand score)         |
| `aggressive`    | Highest average hand score                                 |
| `sure_bet`      | Highest minimum points                                     |
| `risky_bet`     | Highest maximum points                                     |
| `hail_mary`     | Best chance at a high score (high score X+ about 5% of the time) |

Against a random player, `recommended` performed best, followed by `aggressive` and `sure_bet`.

<br>

## Implemented Players
All players inherit from the abstract `BasePlayer` class. Since every round has two decision phases, a player can use a different strategy for each, which is reflected in the player's name:

| Tag  | Phase      | Strategy                                                                    |
|------|------------|-----------------------------------------------------------------------------|
| `DR` | Discarding | **Random** - discards two random cards.                                     |
| `DG` | Discarding | **Greedy** - keeps the hand with the highest immediate score.               |
| `DA` | Discarding | **Analytical** - statistically best discard according to the Discard Evaluator. |
| `DN` | Discarding | **Neural Network** - trained with reinforcement learning.                   |
| `PR` | Pegging    | **Random** - plays a random legal card.                                     |
| `PG` | Pegging    | **Greedy** - plays the card that scores the most points right now.          |
| `PN` | Pegging    | **Neural Network** - trained with reinforcement learning.                   |

The **Show** phase needs no agent — points are calculated automatically by the `Scoring` class.

<br>

## Neural Networks
Instead of using a single network for the entire game, each agent uses **two separate networks**: one for the discarding phase and one for the pegging phase. The two phases involve very different decisions (choosing 2 of 6 cards vs. choosing the order of play), so a network specialized in one task is much easier to train.

### Network Architecture
Networks inherit from `BaseDiscardNet` / `BasePeggingNet` (which extend PyTorch's `nn.Module`). Four structures were tested: **shallow/deep** × **wide/slim**, each with ReLU, LeakyReLU, SELU, Sigmoid, and Tanh activations.

### Training
Two training modes are supported:
* **Coached** - the agent takes the action suggested by a coach (the Analytical player for discarding, the Greedy player for pegging).
* **Non-coached** - the agent takes the action it is most confident about.

#### Using the Discard Trainer
```py
    net = DNT_ShallowSeluWide()
    
    DiscardTrainer.train(
        discard_network = net,
        lr = 0.0001,
        wd = 0.01,
        epochs = 50_000,
        alpha = 1.0  # for coached training, for self-play leave at 0.0
    )
    
    DiscardTrainer.save(net, 'training_name', 'comment')  # Saves training logs by default
```

For training with pregenerated data (states), check out the `DiscardTrainerPreLoaded` class.

#### Using the Pegging Trainer
```py
    net = PNT_ShallowSeluSlim()

    PeggingTrainer.train(
        pegging_network = net,
        lr = 0.0001,
        wd = 0.01,
        epochs = 50_000,
        opponent = DAPGPlayer(),
        alpha = 1.0  # for coached training, for self-play leave at 0.0
    )

    PeggingTrainer.save(net, 'training_name', 'comment')  # Saves training logs by default
```

For training with pregenerated data (states), check out the `PeggingTrainerPreLoaded` class.

<br>

## Results
All results are win rates over 1000 simulated games (using the `Simulator` class).

**Discarding Networks** (vs. Discard Random, with Pegging Random for both players):

| Player                                              | Win Rate |
|-----------------------------------------------------|----------|
| Discard Analytical (recommended) - reference point  | 97.7%    |
| Best network (SELU, Shallow-Wide)                   | 94.2%    |
| Best network vs. its coach (Discard Analytical)     | 36.4%    |

**Pegging Networks** (vs. Pegging Random, with Discard Greedy for both players):

| Player                                              | Win Rate |
|-----------------------------------------------------|----------|
| Pegging Greedy - reference point                    | 62.5%    |
| Best network (ReLU, Shallow-Slim)                   | 58.6%    |
| Best network vs. its coach (Pegging Greedy)         | 43.5%    |

The discarding network plays far better than random, but doesn't reach the analytical player. The pegging networks only learned simple tricks like fifteens and pairs, with no consistent knowledge of runs or blocking the opponent.

<br>

## Fun Facts
This project killed us.
<img wisrc="https://github.com/user-attachments/assets/426b17b9-6d6d-4133-8049-13761dfde99f"/>
