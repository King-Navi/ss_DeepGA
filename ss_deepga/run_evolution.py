import torch.utils.data as data
from DeepGA.Operators import *
from DeepGA.EncodingClass import Encoding
from DeepGA.Decoding import *
from DeepGA.DataReader import *
from DeepGA.DistributedTraining import *
from DeepGA.DeepGA import *

from ss_deepga.utils.show_layout_classes import show_split_distribution
from ss_deepga.stratified_loaders import make_stratified_loaders_v2
from ss_deepga.resources.constants import PATH_TO_CLASSES, CHECKPOINT_DIR, EXECUTION_ID, IMAGE_SIZE

import DeepGA.EncodingClass as enc
import DeepGA.Operators as ops

# Override the catalogs actually used by EncodingClass
enc.FSIZES = [3, 5, 7, 9]
enc.NFILTERS = [8, 16, 32, 64, 128, 256]
enc.PSIZES = [2, 3]
enc.PTYPE = ["max", "avg"]
enc.NEURONS = [16, 32, 64, 128, 256]
# Some versions of Operators rely on these names existing in that module
ops.FSIZES = enc.FSIZES
ops.NFILTERS = enc.NFILTERS
ops.PSIZES = enc.PSIZES
ops.PTYPE = enc.PTYPE
ops.NEURONS = enc.NEURONS

data_dir = PATH_TO_CLASSES


'''Defining DeepGA hyperparameters'''

#Defining learning rate
lr = 1e-4

#Maximun and minimum numbers of layers to initialize networks
min_conv = 2 # 30
max_conv = 8 # 60
min_full = 1
max_full = 3 # 10
max_params = 3e6
train_epochs = 8 # Epochs to train the best individual found by the GA

'''Genetic Algorithm Parameters'''
cr = 0.9   # Crossover rate
mr = 0.6   # Mutation rate
N = 20      # Population size 20 Se mantuvo en 20
T = 30 #30      # Number of generations
t_size = 5 # tournament size 5
w = 0.1    # penalization weight   0.3

chck_dir = CHECKPOINT_DIR  # Root folder for dumping/loading pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (torch): {torch.version.cuda}")
else:
    print("⚠️ Using CPU (no CUDA/GPU detected)")


train_dl, val_dl, test_dl, n_channels, n_classes, out_size, ds, train_idx, val_idx, test_idx = make_stratified_loaders_v2(
    data_dir=data_dir,
    image_size=IMAGE_SIZE,
    batch_size=7,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    num_workers=2,
)

loss_func = torch.nn.NLLLoss() # because CNN returns log_softmax

execution_ID = EXECUTION_ID #Execution ID for checkpoint, change it for a new execution




results, pop, bestind  = deepGA(execution_ID, True, train_epochs = train_epochs, train_dl=train_dl, val_dl=val_dl,  lr=lr,
                       min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full, max_params=max_params,
                       cr=cr, mr=mr, N=N, T=T, t_size=t_size, w=w, device=device, chck_dir=chck_dir,
                       n_channels =  n_channels , n_classes=n_classes, out_size = out_size, loss_func=loss_func)

# training to more epochs
finalEpochs = 150#100
CNNModel = final_evaluation(execution_ID, bestind, train_dl, val_dl, lr, max_params, w, device, finalEpochs, loss_func, chck_dir, n_channels =  n_channels , n_classes=n_classes, out_size = out_size)