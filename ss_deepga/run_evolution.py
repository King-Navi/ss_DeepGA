import torch.utils.data as data
from DeepGA.Operators import *
from DeepGA.EncodingClass import Encoding
from DeepGA.Decoding import *
from DeepGA.DataReader import *
from DeepGA.DistributedTraining import *
from DeepGA.DeepGA import *

from utils.show_layout_classes import show_split_distribution

data_dir = "radiografias_dxs_pulpares/"


'''Defining DeepGA hyperparameters'''
#Convolutional layers
FSIZES = [3, 5, 7, 9] # Odd Sizes Are Preferred
NFILTERS = [8, 16, 32, 64, 128, 256]

#Pooling layers
PSIZES = [2,3] #[2,3,4,5]
PTYPE = ['max', 'avg']

#Fully connected layers
NEURONS = [16, 32, 64, 128, 256] # for big layers

EXECUTION_ID =708

'''Defining DeepGA hyperparameters'''

#Defining learning rate
lr = 1e-4

#Maximun and minimum numbers of layers to initialize networks
min_conv = 2 # 30
max_conv = 30 # 60
min_full = 1
max_full = 6 # 10
max_params = 9e6
train_epochs = 15 # Epochs to train the best individual found by the GA

'''Genetic Algorithm Parameters'''
cr = 0.7   # Crossover rate
mr = 0.5   # Mutation rate
N = 25      # Population size 20 Se mantuvo en 20
T = 35 #30      # Number of generations
t_size = 5 # tournament size 5
w = 0.1    # penalization weight   0.3

chck_dir = 'point/'  # Root folder for dumping/loading pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version (torch): {torch.version.cuda}")
else:
    print("⚠️ Using CPU (no CUDA/GPU detected)")


train_dl, val_dl, test_dl, n_channels, n_classes, out_size, ds, train_idx, val_idx, test_idx = make_stratified_loaders_v2(
    data_dir=data_dir,
    image_size=128,
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    num_workers=2,
)

# opcional ->>> verificar distribución
show_split_distribution(ds, train_idx, "TRAIN")
show_split_distribution(ds, val_idx,   "VAL")
show_split_distribution(ds, test_idx,  "TEST")

loss_func = nn.CrossEntropyLoss()

execution_ID = EXECUTION_ID #Execution ID for checkpoint, change it for a new execution




results, pop, bestind  = deepGA(execution_ID, True, train_epochs = train_epochs, train_dl=train_dl, val_dl=val_dl,  lr=lr,
                       min_conv=min_conv, max_conv=max_conv, min_full=min_full, max_full=max_full, max_params=max_params,
                       cr=cr, mr=mr, N=N, T=T, t_size=t_size, w=w, device=device, chck_dir=chck_dir,
                       n_channels =  n_channels , n_classes=n_classes, out_size = out_size, loss_func=loss_func)

# training to more epochs
finalEpochs = 150#100
CNNModel = final_evaluation(execution_ID, bestind, train_dl, val_dl, lr, max_params, w, device, finalEpochs, loss_func, chck_dir, n_channels =  n_channels , n_classes=n_classes, out_size = out_size)