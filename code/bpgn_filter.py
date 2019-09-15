import numpy as np
import matplotlib.pyplot as plt
import random

# represent a game from .bpgn
import sys


class Game():
    def __init__(self):
        self.event = ''
        self.site = ''
        self.date = ''
        self.time = ''
        self.db_game_no = ''
        self.wh_a = ''
        self.wh_a_player = ''
        self.wh_a_elo = 0
        self.bl_a = ''
        self.bl_a_player = ''
        self.bl_a_elo = 0
        self.wh_b = ''
        self.wh_b_player = ''
        self.wh_b_elo = 0
        self.bl_b = ''
        self.bl_b_player = ''
        self.bl_b_elo = 0
        self.time_ctrl = ''
        self.lag = ''
        self.result = ''
        self.note = ''
        self.moves = ''
        self.end = ''

    # get the lowest ELO of 4 players in a game
    def get_min_elo(self):
        return min([self.wh_a_elo, self.bl_a_elo, self.wh_b_elo, self.bl_b_elo])


# create a list of all games
def create_games(path):
    games = []
    file = open(path, encoding="iso-8859-1")
    content = file.read().splitlines()

    # initialize games
    for line in content:
        if '[Event ' in line:
            game = Game()
            games.append(game)

    current_game = 0
    for line in content:
        if '[Event ' in line:
            game = games[current_game]
            setattr(game, 'event', line)
            current_game += 1

        if '[Site ' in line:
            setattr(game, 'site', line)
            continue

        if '[Date ' in line:
            setattr(game, 'date', line)
            continue

        if '[Time ' in line:
            setattr(game, 'time', line)
            continue

        if '[BughouseDBGameNo ' in line:
            setattr(game, 'db_game_no', line)
            continue

        if '[WhiteA ' in line:
            setattr(game, 'wh_a', line)
            split = line.split('"')[1]
            setattr(game, 'wh_a_player', split)
            split = line.split('"')[3]
            if split is not '':
                setattr(game, 'wh_a_elo', int(split))
            continue

        if '[BlackA ' in line:
            setattr(game, 'bl_a', line)
            split = line.split('"')[1]
            setattr(game, 'bl_a_player', split)
            split = line.split('"')[3]
            if split is not '':
                setattr(game, 'bl_a_elo', int(split))
            continue

        if '[WhiteB ' in line:
            setattr(game, 'wh_b', line)
            split = line.split('"')[1]
            setattr(game, 'wh_b_player', split)
            split = line.split('"')[3]
            if split is not '':
                setattr(game, 'wh_b_elo', int(split))
            continue

        if '[BlackB ' in line:
            setattr(game, 'bl_b', line)
            split = line.split('"')[1]
            setattr(game, 'bl_b_player', split)
            split = line.split('"')[3]
            if split is not '':
                setattr(game, 'bl_b_elo', int(split))
            continue

        if '[TimeControl ' in line:
            setattr(game, 'time_ctrl', line)
            continue

        if '[Lag ' in line:
            setattr(game, 'lag', line)
            continue

        if '[Result ' in line:
            setattr(game, 'result', line)
            continue

        if '{C:' in line:
            setattr(game, 'note', line)
            continue

        if '1A. ' in line or '1B. ' in line:
            setattr(game, 'moves', line)
            continue

        if '{Game aborted' in line or 'checkmated}' in line or 'resigns}' in line:
            setattr(game, 'end', line)
            continue

    return games


# create a list of valid games (non-aborted) only
def create_valid_games(path):
    games = create_games(path)
    valid_games = [game for game in games if not ('aborted' in game.end)]

    return valid_games


# crate a list of valid games from all .bpgn in root directory
def create_total_games(files):
    total_games = []
    for path in files:
        valid_games = create_valid_games(path)
        total_games.extend(valid_games)

    return total_games


# create a list of minimal ELOs from games
def create_min_elos(games):
    min_elos = []
    for game in games:
        min_elos.append(game.get_min_elo())

    return min_elos


# export .bpgn containing games in top percentile
def export_bpgn(files, percentile):
    total_games = create_total_games(files)
    # sort
    total_games.sort(key=lambda x: x.get_min_elo(), reverse=True)
    # pick best games
    total_number_of_games = int(len(total_games) * percentile / 100)
    number_of_validation_games =  int(total_number_of_games * 0.02)
    top_games = total_games[0:total_number_of_games]
    random.shuffle(top_games)
    validation_set = top_games[0:number_of_validation_games+1]
    trainig_set = top_games[number_of_validation_games+1:]
    # output .bpgn
    dest_path = 'validation_' + 'top_' + str(percentile) + '_percent.bpgn'
    file = open(dest_path, 'w')
    for game in validation_set:
        file.write(game.event + '\n')
        file.write(game.site + '\n')
        file.write(game.date + '\n')
        file.write(game.time + '\n')
        file.write(game.db_game_no + '\n')
        file.write(game.wh_a + '\n')
        file.write(game.bl_a + '\n')
        file.write(game.wh_b + '\n')
        file.write(game.bl_b + '\n')
        file.write(game.time_ctrl + '\n')
        file.write(game.lag + '\n')
        file.write(game.result + '\n')
        file.write('\n')
        file.write(game.note + '\n')
        file.write(game.moves + '\n')
        file.write(game.end + '\n')
        file.write('\n')

    dest_path = 'training_' + 'top_' + str(percentile) + '_percent.bpgn'
    file = open(dest_path, 'w')
    for game in trainig_set:
        file.write(game.event + '\n')
        file.write(game.site + '\n')
        file.write(game.date + '\n')
        file.write(game.time + '\n')
        file.write(game.db_game_no + '\n')
        file.write(game.wh_a + '\n')
        file.write(game.bl_a + '\n')
        file.write(game.wh_b + '\n')
        file.write(game.bl_b + '\n')
        file.write(game.time_ctrl + '\n')
        file.write(game.lag + '\n')
        file.write(game.result + '\n')
        file.write('\n')
        file.write(game.note + '\n')
        file.write(game.moves + '\n')
        file.write(game.end + '\n')
        file.write('\n')


# plot ELOs distribution based on groups
def draw_min_elos(files, offset):
    min_elos=create_min_elos(create_total_games(files))

    # max value in x-axis
    max_of_min_elos = max(min_elos)

    # group labels on x-axis
    labels = []

    # how tall columns will be
    label_values = []

    # divide min_elos into smaller groups with offset:
    # e.g: min_elos range (0- 2550), offset= 400: groups are: 0-400, 400-800,..., 2000-2400, 2400-2550
    for i in range(0, np.math.ceil(max_of_min_elos / offset)):

        # create labels under the columns
        if ((i + 1) * offset > max_of_min_elos):
            labels.append(str(i * offset) + '\n' + '-' + str(max_of_min_elos))
        else:
            labels.append(str(i * offset) + '\n' + '-' + str((i + 1) * offset))

        # save values (in percentage)
        temp = []
        for min_elo in min_elos:
            if (i * offset <= min_elo) and (min_elo < (i + 1) * offset):
                temp.append(min_elo)
        label_values.append(len(temp) / len(min_elos) * 100)

    # start drawing
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, label_values, align='center', alpha=1)
    plt.xticks(y_pos, labels)
    plt.ylabel('Percentage of games')
    plt.xlabel('Minimal ELO')
    plt.title('Distribution of games based on minimal ELO')
    plt.show()


# Test
# draw_min_elos(sys.argv[1:], 20)
export_bpgn(sys.argv[1:], 10)