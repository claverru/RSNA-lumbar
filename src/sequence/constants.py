from src import constants


N_IMG_ENSEMBLES = 5

M = 103
D = 8

N_LEVEL = len(constants.LEVELS)
N_CONDITION_LEVEL = len(constants.CONDITION_LEVEL)
N_SEVERITY = len(constants.SEVERITY2LABEL)

F = 1920
SIMILARITY_DIM = 16

INPUT_SIZE = (F + SIMILARITY_DIM * N_LEVEL + N_CONDITION_LEVEL * N_SEVERITY) * N_IMG_ENSEMBLES + M + D
