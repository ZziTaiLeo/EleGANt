from fvcore.common.config import CfgNode
_C = CfgNode()
_C.PGT = CfgNode()
_C.PGT.EYE_MARGIN = 12
_C.PGT.LIP_MARGIN = 4
_C.PGT.ANNEALING = True
_C.PGT.SKIN_ALPHA = 0.3
_C.PGT.SKIN_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.SKIN_ALPHA_VALUES = (0.2, 0.4, 0.3, 0.2)
_C.PGT.EYE_ALPHA = 0.8
_C.PGT.EYE_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.EYE_ALPHA_VALUES = (0.6, 0.8, 0.6, 0.4)
_C.PGT.LIP_ALPHA = 0.1
_C.PGT.LIP_ALPHA_MILESTONES = (0, 12, 24, 50)
_C.PGT.LIP_ALPHA_VALUES = (0.05, 0.2, 0.1, 0.0)

# Data settings
_C.DATA = CfgNode()
_C.DATA.NUM_WORKERS = 4
_C.DATA.BATCH_SIZE = 1
_C.DATA.IMG_SIZE = 512

# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS = [12, 13]
_C.PREPROCESS.FACE_CLASS = [1, 10]
_C.PREPROCESS.EYEBROW_CLASS = [2, 3]
_C.PREPROCESS.EYE_CLASS = [4, 5]
_C.PREPROCESS.LANDMARK_POINTS = 68

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.LAMBDA_MAKEUP = 100
_C.LOSS.LAMBDA_SKIN = 0.1
_C.LOSS.LAMBDA_EYE = 1.5
_C.LOSS.LAMBDA_LIP = 1
_C.LOSS.LAMBDA_MAKEUP_LIP = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_LIP
_C.LOSS.LAMBDA_MAKEUP_SKIN = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_SKIN
_C.LOSS.LAMBDA_MAKEUP_EYE = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_EYE

# Training hyper-parameters
_C.TRAINING = CfgNode()
_C.TRAINING.G_LR = 2e-4
_C.TRAINING.D_LR = 2e-4
_C.TRAINING.BETA1 = 0.5
_C.TRAINING.BETA2 = 0.999
_C.TRAINING.NUM_EPOCHS = 50
_C.TRAINING.LR_DECAY_FACTOR = 5e-2
_C.TRAINING.DOUBLE_D = False

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.LAMBDA_A = 10.0
_C.LOSS.LAMBDA_B = 10.0
_C.LOSS.LAMBDA_IDT = 0.5
_C.LOSS.LAMBDA_REC = 10
_C.LOSS.LAMBDA_MAKEUP = 100
_C.LOSS.LAMBDA_SKIN = 0.1
_C.LOSS.LAMBDA_EYE = 1.5
_C.LOSS.LAMBDA_LIP = 1
_C.LOSS.LAMBDA_MAKEUP_LIP = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_LIP
_C.LOSS.LAMBDA_MAKEUP_SKIN = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_SKIN
_C.LOSS.LAMBDA_MAKEUP_EYE = _C.LOSS.LAMBDA_MAKEUP * _C.LOSS.LAMBDA_EYE
_C.LOSS.LAMBDA_VGG = 5e-3

def get_config() -> CfgNode:
    return _C
