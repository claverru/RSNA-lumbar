from src import constants


N_IMG_ENSEMBLES = 5

M = 103
P = len(constants.DESCRIPTIONS)

N_LEVEL = len(constants.LEVELS)
N_CONDITION_LEVEL = len(constants.CONDITION_LEVEL)
N_SEVERITY = len(constants.SEVERITY2LABEL) + 1
N_CONDITION = len(constants.CONDITIONS_COMPLETE)
XY = 2
Z = 1

F = 1920
SIMILARITY_DIM = 64

LEVEL_F = N_LEVEL * SIMILARITY_DIM
OUT_F = N_CONDITION_LEVEL * N_SEVERITY
XY_F = N_CONDITION_LEVEL * XY

INPUT_SIZE = (F + LEVEL_F + OUT_F + XY_F + Z + P) * N_IMG_ENSEMBLES + M + P


# def get_masks():
#     masks = []
#     for i in range(5):
#         levels = np.zeros((N_LEVEL, 1), dtype=bool)
#         levels[i] = True
#         levels = levels.repeat(SIMILARITY_DIM).tolist()
#         mask = (F * [False] + levels + IMG_O2 * [False] + IMG_O3 * [False] + P * [False]) * N_IMG_ENSEMBLES + M * [True] + P * [True]
#         masks.append(mask)
#     return masks
