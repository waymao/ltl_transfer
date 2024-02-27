# LTL-Transfer
This work shows ways to reuse policies trained to solve a set of training tasks, specified by linear temporal logic (LTL), to solve novel LTL tasks in a zero-shot manner.
Please see the following paper for more details.

TODO add paper

## Installation instructions
```
mamba create -n <name> numpy h5py python=3.10 scipy
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Running Experiments
Please Run the following commands to run the experiment. 

### Step 0. Set up the policy bank location
Create a folder at `$HOME/data/shared/ltl-transfer-ts` where `$HOME` is your home folder.
You may change the file location as needed, but make sure to update the save_dpath parameter in every
command below.

Other common parameters:

| parameter name | options | description |
| -------------- | ------- | ----------- |
| `--train_type`   | 'sequence', 'test_until', 'interleaving', 'safety', 'hard', 'mixed', 'soft_strict', 'soft', 'no_orders', 'individual' | The training LTL dataset to be used |
| `--test_type`    | 'hard', 'mixed', 'soft_strict', 'soft', 'no_orders' | The testing LTL dataset to be used |
| `--map`          | medium: 21, 22, 23, 24; small toy problem: 13; large: 32 | the map id |
| `--prob`         | 1.0, 0.9, 0.8 | probability of intended action succeeding. For Miniworld, prob 1.0 will turn off drifting while 0.9 will keep drifting on. see `envs/miniworld/constants` for more information. |
| `--run_id`       | any integer   | the seed used in the experiments  |
| `--save_dpath`   | a path        | the path of the saved policybank and logs|
| `--domain_name`  | 'minecraft', 'spot' | the name of the dataset with landmarks specialized in each domain. |
| `--device`       | 'cpu', 'cuda'       | device to run the NN    |
| `--game_name`    | grid, miniworld, miniworld_simp_no_vis, miniworld_simp_lidar | type of game env to run |
| `--run_subfolder`| subfolder name  | custom subfolder under the policybank folder (used to store different runs of tuning) |
| `--rl_algo`      | 'sac', 'ppo'(to be worked on) | rl algorithm used  | 

### Step 1. Initialize policy banks
```
PYGLET_HEADLESS=true python3 init_ts_policy_bank.py --train_size 50 \
    --rl_algo dsac --map 21 --domain_name spot --prob=1.0 \
    --game_name miniworld_simp_no_vis --train_type mixed \
    --save_dpath=$HOME/data/shared/ltl-transfer-ts
```

### Step 2. Train the policy individually
Replace `{}` below with the desired LTL id to train.
```
PYGLET_HEADLESS=true python3 run_ts_single_policy.py \
      --train_size 50 --rl_algo dsac --map 21 --ltl_id {} \
      --game_name miniworld_simp_no_vis --train_type mixed \
      --save_dpath=$HOME/data/shared/ltl-transfer-ts
```

### Step 3. Run Rollout of the policies
```
PYGLET_HEADLESS=true python run_ts_single_rollout.py \
      --save_dpath=$HOME/data/shared/ltl-transfer-ts \
      --game_name miniworld_simp_no_vis --map 21 \
      --train_type mixed --ltl {} --no_deterministic_eval
```
Additional parameters:
| parameter name | options | description |
| -------------- | ------- | ----------- |
| `--no_deterministic_eval`   | True/False | If present, sample an action according to the distribution. If not, use argmax to find the action |
| `--relabel_seed` | integer | the seed used by the sampler for relabeling |
| `--rollout_method` | 'uniform', 'random' | Rollout method |

### Step 4. Run transfer
```
PYGLET_HEADLESS=true python run_ts_transfer.py \
    --save_dpath=$HOME/data/shared/ltl-transfer-ts \
    --game_name miniworld_simp_no_vis --map 21 \
    --train_type mixed --task_id $LTL_ID -v \
    --relabel_method knn_random --relabel_seed 0
```
Additional parameters:
| parameter name | options | description |
| -------------- | ------- | ----------- |
| `--relabel_method`   | '{knn,radius}_{random,uniform}', ' | First part is the matching method, knn or all points in a radius. Second part specifies the relabeling method to be used as data source. e.g. knn_random |
| `--task_id`          | integer | The id of the task in the testing set to be run. |
| `-v`                 | True/False | Whether to print all details of transfer. (if not present, only a JSON summary for each episode will be printed)| 

## Issues Note
### Miniworld Rendering
Miniworld is a game environment built on OpenGL. It is necessary to have some graphics card and a display, even when running the simplified non-visual observation environments. If running remotely, pyglet to run in headless mode, hence we need to add "PYGLET_HEADLESS=true" to the commands.

You may also use PYGLET_HEADLESS_DEVICE={num} to select the desired GPU
if you have multiple.
