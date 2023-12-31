from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from tinydb import TinyDB
from pathos.multiprocessing import ProcessPool
import util
from glicko2.glicko2 import Glicko2
import board as blokusBoard
from simulator import BlokusSim
from util import get_timestamp
from tqdm import tqdm
import itertools
import random

from players import Player
from players.SmallestFirst import SmallestFirstPlayer
from players.AimCenter import AimCenterPlayer
from players.AvoidCenter import AvoidCenterPlayer
from players.BiggestFirst import BiggestFirstPlayer
from players.Random import RandomPlayer
from players.BigEdge import BigEdgePlayer
from players.AliceInWonderland import AliceInWonderlandPlayer
from players.BigCenter import BigCenterPlayer
from players.BobInLife import BobInLifePlayer
from players.CharlieInConfusion import CharlieInConfusion
from players.DianaInJungle import DianaInJunglePlayer
from players.EggHeadInFridge import EggHeadInFridgePlayer
from players.FredrickIsPompus import FredrickIsPompusPlayer

PLOT_GLICKO_OVER_TIME = True
NUMBER_OF_MATCHES = 1000
NUMBER_OF_PARALLEL_PROCESSES = 16
BATCH_COMMIT = 1000
PLOT_LAST_POINTS = 500


def get_players():
    return [
        AimCenterPlayer,
        AvoidCenterPlayer,
        BiggestFirstPlayer,
        RandomPlayer,
        SmallestFirstPlayer,
        BigCenterPlayer,
        AliceInWonderlandPlayer,
        BobInLifePlayer,
        CharlieInConfusion,
        DianaInJunglePlayer,
        EggHeadInFridgePlayer,
        FredrickIsPompusPlayer,
        BigEdgePlayer
    ]


def play_match(player1_class: type(Player), player2_class: type(Player)):
    board = blokusBoard.BlokusBoard()
    players = [player1_class(board), player2_class(board)]
    sim = BlokusSim(board, players)
    sim.run_steps(21 * 4)
    score = sim.get_current_score()
    player_steps = sim.player_steps
    # remaining_players = sim.get_number_of_remaining_pieces()
    data = {
        "player1_id": players[0].get_player_id(),
        "player2_id": players[1].get_player_id(),
        "player1_score": int(score[0]),
        "player2_score": int(score[1]),
        "global_steps": int(sim.step),
        "player1_steps": int(player_steps[0]),
        "player2_steps": int(player_steps[1]),
        # "player1_remaining_pieces": int(remaining_players[0]), # fix not working rn
        # "player2_remaining_pieces": int(remaining_players[1]),
        "time_stamp": get_timestamp(),
        "match_id": sim.game_id
    }
    return data


def update_rating(glicko, id_dict, p1_id, p2_id, p1_score, p2_score):
    drawn = p1_score == p2_score
    if p1_score > p2_score:
        id_dict[p1_id], id_dict[p2_id] = glicko.rate_1vs1(id_dict[p1_id], id_dict[p2_id], drawn)
    else:
        id_dict[p1_id], id_dict[p2_id] = glicko.rate_1vs1(id_dict[p1_id], id_dict[p2_id], drawn)


def main():
    db = TinyDB('match_replays/played_matches.json')
    players = get_players()
    players_ids = []
    players_id_to_class = {}
    player_id_to_rating = {}
    player_id_to_history = {}

    glicko = Glicko2()

    temp_board = blokusBoard.BlokusBoard()
    for player in players:
        player_id = player(temp_board).get_player_id()
        players_ids.append(player_id)
        players_id_to_class[player_id] = player
        player_id_to_rating[player_id] = glicko.create_rating()
        player_id_to_history[player_id] = []

    match_counts = {}  # Set(players) -> number_of_matches_played
    for combo in itertools.combinations(players_ids, 2):
        match_counts[frozenset(combo)] = 0

    player1_required = []
    player2_required = []

    for match in tqdm(iter(db), desc="Counting old matches", total=len(db)):
        p1_id = match["player1_id"]
        p2_id = match["player2_id"]
        match_counts[frozenset((p1_id, p2_id))] += 1

    for combo, played_matches in match_counts.items():
        required_matches = NUMBER_OF_MATCHES - played_matches
        players = list(combo)
        player1_required.extend([players_id_to_class[players[0]]] * required_matches)
        player2_required.extend([players_id_to_class[players[1]]] * required_matches)

    if len(player1_required) > 0:
        player1_required, player2_required = util.shuffle_together(player1_required, player2_required)

        processPool = ProcessPool(nodes=NUMBER_OF_PARALLEL_PROCESSES)
        results = processPool.imap(play_match, player1_required, player2_required)

        batch = []
        idx = 0

        for match in tqdm(results, desc="Playing matches", total=len(player1_required)):
            if idx > BATCH_COMMIT:
                db.insert_multiple(batch)
                batch = []
                idx = 0
            batch.append(match)
        if len(batch) > 0:
            db.insert_multiple(batch)

    all_matches = db.all()
    random.shuffle(all_matches)

    for match in tqdm(all_matches, desc="Rating all matches"):
        p1_id = match["player1_id"]
        p2_id = match["player2_id"]
        match_counts[frozenset((p1_id, p2_id))] += 1
        update_rating(
            glicko,
            player_id_to_rating,
            p1_id,
            p2_id,
            match["player1_score"],
            match["player2_score"],
        )
        if PLOT_GLICKO_OVER_TIME:
            player_id_to_history[p1_id].append(player_id_to_rating[p1_id].mu)
            player_id_to_history[p2_id].append(player_id_to_rating[p2_id].mu)

    players_ids = sorted(players_ids, key=lambda pid: player_id_to_rating[pid].mu)

    if PLOT_GLICKO_OVER_TIME:
        fig, ax = plt.subplots()
        cm = plt.get_cmap('gist_rainbow')
        num_players = len(players_ids)
        ax.set_prop_cycle(color=[cm(1. * i / num_players) for i in range(num_players)])
    for player_id in players_ids:
        print(f"{players_id_to_class[player_id].__name__} -> {player_id_to_rating[player_id].mu}")
        if PLOT_GLICKO_OVER_TIME:
            data_to_plot = player_id_to_history[player_id]
            how_many_to_plot = PLOT_LAST_POINTS if len(data_to_plot) > PLOT_LAST_POINTS else len(data_to_plot)
            data_to_plot = data_to_plot[-how_many_to_plot:]
            plt.plot(list(range(len(data_to_plot))), data_to_plot,
                     label=str(players_id_to_class[player_id].__name__), linewidth='0.5')
    if PLOT_GLICKO_OVER_TIME:
        plt.legend(loc='lower left', fontsize="7")
        plt.show()


if __name__ == '__main__':
    main()
