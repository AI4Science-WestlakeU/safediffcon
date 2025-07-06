#!/usr/bin/env python

import os
import sys, os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings at import time
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np

np.random.seed(0)

from scipy import interpolate
from common.model_structure import *
from common.wall import *
from common.setting import *
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from copy import deepcopy

import pickle as pkl

# Setting
wide = True
base_path = os.path.abspath(os.path.dirname(sys.argv[0]))
os.makedirs(os.path.join(base_path, 'images'), exist_ok=True)
kstar_img_path = os.path.join(base_path, 'images', 'insideKSTAR.jpg')
max_models = 10
init_models = 1
max_shape_models = 4
seq_len = 10
decimals = np.log10(1000)
dpi = 1
plot_length = 40
t_delay = 0.05
steady_model = False
lookback = 3
show_inputs = False

n_model_box = 1

rcParamsSetting(dpi)

# Fixed setting
year_in = 2021
ec_freq = 105.e9

# Path of weights
lstm_model_path = os.path.join(base_path, 'weights', 'lstm', 'v220505')
nn_model_path = os.path.join(base_path, 'weights', 'nn')
bpw_model_path = os.path.join(base_path, 'weights', 'bpw')
k2rz_model_path = os.path.join(base_path, 'weights', 'k2rz')
rl_model_path = os.path.join(
    base_path, 
    'weights', 
    'rl', 
    'rt_control', 
    '3frame_v220505', 
    'best_model.zip', 
)


# RL setting
low_target = [0.8, 4.0, 0.80]
high_target = [2.1, 7.0, 1.05]
low_action = [0.3, 0.0, 0.0, 0.0, 1.6, 0.15, 0.5, 1.265, 2.14]
high_action = [0.8, 1.75, 1.75, 1.5, 1.95, 0.5, 0.85, 1.36, 2.3]

low_state = (low_action + low_target) * lookback + low_target
high_state = (high_action + high_target) * lookback + high_target

# Inputs
input_params = [
    'Ip [MA]', 'Bt [T]', 'GW.frac. [-]',
    'Pnb1a [MW]', 'Pnb1b [MW]', 'Pnb1c [MW]',
    'Pec2 [MW]', 'Pec3 [MW]', 'Zec2 [cm]', 'Zec3 [cm]',
    'In.Mid. [m]', 'Out.Mid. [m]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]'
]
input_mins = [0.3, 1.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -10, -10, 1.265, 2.18, 1.6, 0.1, 0.5]
input_maxs = [0.8, 2.7, 0.6, 1.75, 1.75, 1.5, 0.8, 0.8, 10, 10, 1.36, 2.29, 2.0, 0.5, 0.9]
input_init = [0.5, 1.8, 0.33, 1.5, 1.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.32, 2.22, 1.7, 0.3, 0.75]

# Outputs
output_params0 = ['βn', 'q95', 'q0', 'li']
output_params1 = ['βp', 'wmhd']
output_params2 = ['βn', 'βp', 'h89', 'h98', 'q95', 'q0', 'li', 'wmhd'] # for all the outputs
dummy_params = [
    'Ip [MA]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]', 
    'In.Mid. [m]', 'Out.Mid. [m]', 'Pnb1a [MW]', 'Pnb1b [MW]', 'Pnb1c [MW]'
]

# Targets
target_params = ['βp', 'q95', 'li']
target_mins = [0.8, 4.0, 0.80]
target_maxs = [2.1, 7.0, 1.05]
target_init = [1.45, 5.5, 0.925]


rand_target_mins = [1.06, 4.6, 0.85]
rand_target_maxs = [1.84, 6.4, 1.00]

# =======================
# Utility Functions
# =======================

def i2f(i, decimals=decimals):
    """Convert integer to float with fixed decimal precision."""
    return float(i/ (10 ** decimals))

def f2i(f, decimals=decimals):
    """Convert float to integer by scaling."""
    return int(f * (10 ** decimals))

# =======================
# Data Generator Class
# =======================

class KSTARSolver:
    def __init__(self, sample_id=0):
        # Initial condition
        np.random.seed(sample_id)
        self.first = True
        self.time = np.linspace(-0.1 * (plot_length - 1), 0, plot_length)
        self.outputs = {p: [0.0] for p in output_params2}
        self.dummy = {p: [0.0] for p in dummy_params}
        self.x = np.zeros([seq_len, 18])
        self.new_action = np.array(low_action)
        self.histories = [list(low_action) + list(target_init)] * lookback
        self.img = plt.imread(kstar_img_path)
        
        # Load models
        if steady_model:
            self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=max_models)
        else:
            self.kstar_nn = kstar_nn(model_path=nn_model_path, n_models=1)
            self.kstar_lstm = kstar_v220505(model_path=lstm_model_path, n_models=max_models)
        self.k2rz = k2rz(model_path=k2rz_model_path, n_models=max_shape_models)
        self.bpw_nn = bpw_nn(model_path=bpw_model_path, n_models=max_models)

        self.reset_model_number()
        
        self.vis_data_dir = './vis_data/sample_{}'.format(sample_id)
        
        os.makedirs('./vis_data', exist_ok=True)
        os.makedirs('./vis_data/sample_{}'.format(sample_id), exist_ok=True)
        
        # Initialize inputs
        self.initialize_inputs()


    def initialize_inputs(self):
        """Initialize input parameters scaled as integers."""
        self.inputs = {param: f2i(val) for param, val in zip(input_params, input_init)}
        
        
    def reset_model_number(self):
        """ MImic the original codes behavior """
        if steady_model:
            self.kstar_nn.nmodels = n_model_box
        else:
            self.kstar_lstm.nmodels = n_model_box
        self.bpw_nn.nmodels = n_model_box
        
    def predict_0d(self, steady=True):
        """Predict 0D plasma parameters."""
        
        # input_params = [
        #     'Ip [MA]', 'Bt [T]', 'GW.frac. [-]',
        #     'Pnb1a [MW]', 'Pnb1b [MW]', 'Pnb1c [MW]',
        #     'Pec2 [MW]', 'Pec3 [MW]', 'Zec2 [cm]', 'Zec3 [cm]',
        #     'In.Mid. [m]', 'Out.Mid. [m]', 'Elon. [-]', 'Up.Tri. [-]', 'Lo.Tri. [-]'
        # ]
        
        if steady:
            x = np.zeros(17)
            
            # pick the right input parameters
            idx_convert = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 10, 2]
            for i in range(len(x) - 1):
                param = input_params[idx_convert[i]]
                x[i] = i2f(self.inputs[param])
            
            # x[: -1] -> [Ip, Bt, Pnb1a, Pnb1b, Pnb1c,
            #       Pec2, Pec3, Zec2, Zec3,
            #       In.Mid[1]., Out.Mid., Elon., Up.Tri., Lo.Tri.,
            #       In.Mid[2]., GW.frac.]
            
            # Handle special cases
            # In.Mid[1], Out.Mid = 0.5 * (In.Mid + Out.Mid), 0.5 * (Out.Mid - In.Mid)
            x[9], x[10] = 0.5 * (x[9] + x[10]), 0.5 * (x[10] - x[9])
            
            # In.Mid[2] = 1.0 if In.Mid > 1.265 else 0.0
            x[14] = 1.0 if x[14] > 1.265 + 1.e-4 else 0.0
            
            x[-1] = year_in
            
            # y -> [βn, q95, q0, li]
            y = self.kstar_nn.predict(x)
            
            # Update the outputs
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])
                
            # Update the 0 index input of LSTM model, shape [seq_len=10, 18]
            self.x[:, :len(output_params0)] = y
            idx_convert = [0, 1, 2, 12, 13, 14, 10, 11, 3, 4, 5, 6, 10]
            for i in range(len(self.x[0]) - 1 - 4):
                param = input_params[idx_convert[i]]
                self.x[:, i + 4] = i2f(self.inputs[param])
            
            # self.x[:-1] -> [βn, q95, q0, li,
            #            Ip, Bt, GW.frac.,
            #            Elon., Up.Tri., Lo.Tri., 
            #            In.Mid., Out.Mid.,
            #            Pnb1a, Pnb1b, Pnb1c,
            #            Pec2, In.Mid.]
            
            # Pec2 = Pec2 + Pec3
            self.x[:, 11 + 4] += i2f(self.inputs[input_params[7]])
            
            # In.Mid = 1.0 if In.Mid[last step] > 1.265 else 0.0
            self.x[:, 12 + 4] = 1.0 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0.0
            self.x[:, -1] = year_in
            
        else:
            # shift the input buffer conditions of LSTM model
            # self.x shape [seq_len=10, 18]
            self.x[:-1, len(output_params0):] = self.x[1:, len(output_params0):]
            
            # pick the right input parameters
            idx_convert = [0, 1, 2, 12, 13, 14, 10, 11, 3, 4, 5, 6, 10]
            for i in range(len(self.x[0]) - 1 - 4):
                param = input_params[idx_convert[i]]
                self.x[-1, i + 4] = i2f(self.inputs[param])
            
            # self.x[:-1] -> [βn, q95, q0, li,
            #            Ip, Bt, GW.frac.,
            #            Elon., Up.Tri., Lo.Tri., 
            #            In.Mid., Out.Mid.,
            #            Pnb1a, Pnb1b, Pnb1c,
            #            Pec2, In.Mid.]
            
            # Pec2 = Pec2 + Pec3
            self.x[-1, 11 + 4] += i2f(self.inputs[input_params[7]])
            
            # In.Mid = 1.0 if In.Mid[last step] > 1.265 else 0.0
            self.x[-1, 12 + 4] = 1.0 if self.x[-1, 12 + 4] > 1.265 + 1.e-4 else 0.0 
            
            # get the output from the LSTM model
            y = self.kstar_lstm.predict(self.x)
            
            # shift the input buffer states [βn, q95, q0, li] of LSTM model
            self.x[:-1, :len(output_params0)] = self.x[1:, :len(output_params0)]
            self.x[-1, :len(output_params0)] = y

            # Update the outputs            
            for i in range(len(output_params0)):
                if len(self.outputs[output_params0[i]]) >= plot_length:
                    del self.outputs[output_params0[i]][0]
                elif len(self.outputs[output_params0[i]]) == 1:
                    self.outputs[output_params0[i]][0] = y[i]
                self.outputs[output_params0[i]].append(y[i])
                
        # Predict output_params1 (βp, wmhd)
        x = np.zeros(8)
        idx_convert = [0, 0, 1, 10, 11, 12, 13, 14]
        x[0] = self.outputs['βn'][-1]
        for i in range(1, len(x)):
            param = input_params[idx_convert[i]]
            x[i] = i2f(self.inputs[param])
            
        # x -> [βn, Ip, Bt, In.Mid, Out.Mid, Elon, Up.Tri, Lo.Tri]
            
        # Handle special cases
        # In.Mid, Out.Mid = 0.5 * (In.Mid + Out.Mid), 0.5 * (Out.Mid - In.Mid)
        x[3], x[4] = 0.5 * (x[3] + x[4]), 0.5 * (x[4] - x[3])
        
        # output the (βp, wmhd)
        y = self.bpw_nn.predict(x)
        
        # Update the outputs
        for i in range(len(output_params1)):
            if len(self.outputs[output_params1[i]]) >= plot_length:
                del self.outputs[output_params1[i]][0]
            elif len(self.outputs[output_params1[i]]) == 1:
                self.outputs[output_params1[i]][0] = y[i]
            self.outputs[output_params1[i]].append(y[i])
        
        # Store dummy parameters
        for p in dummy_params:
            if len(self.dummy[p]) >= plot_length:
                del self.dummy[p][0]
            elif len(self.dummy[p]) == 1:
                self.dummy[p][0] = self.inputs[p]
            self.dummy[p].append(self.inputs[p])
        
        # =======================
        # =======================
        # =======================
        
        # ***********************
        # IMPORTANT: Update the history buffer 
        # history buffer in shape [40, 9 + 3]
        # 9 for actions, 3 for current state
        # ***********************
        
        self.histories[:-1] = self.histories[1:]
        self.histories[-1] = list(self.new_action) + list([
            self.outputs['βp'][-1], 
            self.outputs['q95'][-1], 
            self.outputs['li'][-1]
        ])

        # =======================
        #
        # Following codes only used for visualizations
        #
        # =======================

        # Estimate H factors (h89, h98)
        ip = i2f(self.inputs['Ip [MA]'])
        bt = i2f(self.inputs['Bt [T]'])
        fgw = i2f(self.inputs['GW.frac. [-]'])
        ptot = max(
            i2f(self.inputs['Pnb1a [MW]']) +
            i2f(self.inputs['Pnb1b [MW]']) +
            i2f(self.inputs['Pnb1c [MW]']) +
            i2f(self.inputs['Pec2 [MW]']) +
            i2f(self.inputs['Pec3 [MW]']),
            1.e-1
        )  # Prevent division by zero
        rin = i2f(self.inputs['In.Mid. [m]'])
        rout = i2f(self.inputs['Out.Mid. [m]'])
        k = i2f(self.inputs['Elon. [-]'])
        rgeo, amin = 0.5 * (rin + rout), 0.5 * (rout - rin)
        ne = fgw * 10 * (ip / (np.pi * amin**2))
        m = 2.0  # Mass number

        tau89 = 0.038 * ip**0.85 * bt**0.2 * ne**0.1 * ptot**-0.5 * rgeo**1.5 * k**0.5 * (amin / rgeo)**0.3 * m**0.5
        tau98 = 0.0562 * ip**0.93 * bt**0.15 * ne**0.41 * ptot**-0.69 * rgeo**1.97 * k**0.78 * (amin / rgeo)**0.58 * m**0.19
        h89 = 1.e-6 * self.outputs['wmhd'][-1] / ptot / tau89
        h98 = 1.e-6 * self.outputs['wmhd'][-1] / ptot / tau98
        
        if len(self.outputs['h89']) >= plot_length:
            del self.outputs['h89'][0], self.outputs['h98'][0]
        elif len(self.outputs['h89']) == 1:
            self.outputs['h89'][0], self.outputs['h98'][0] = h89, h98

        self.outputs['h89'].append(h89)
        self.outputs['h98'].append(h98)
        
        # =======================
        # =======================
        # =======================

    def control(self, action):
        """Use actions to update inputs."""
        self.new_action = action
        
        # Actions in 
        # Ip [MA]
        # Pnb1a [MW]
        # Pnb1b [MW]
        # Pnb1c [MW]
        # Elon. [-]
        # Up.Tri. [-]
        # Lo.Tri. [-]
        # In.Mid. [m]
        # Out.Mid. [m]
        
        idx_convert = [0, 3, 4, 5, 12, 13, 14, 10, 11]
        for i, idx in enumerate(idx_convert):
            param = input_params[idx]
            # Ensure the new action is within the action bounds
            action_value = np.clip(self.new_action[i], low_action[i], high_action[i])
            self.inputs[param] = f2i(action_value)

    def return_outputs(self):
        # βn, βp, h89, h98, q95, q0, li, wmhd
        output = {}
        for param in self.outputs.keys():
            output[param] = deepcopy(self.outputs[param])
        return output
    
    def return_controls(self):
        controls = {}
        for param in self.outputs.keys():
            if param in ["βp", "q95", "li"]:
                controls[param] = deepcopy(self.outputs[param][-1])
        return controls
    
    def return_inputs(self):
        # Ip, Pnb1a, Pnb1b, Pnb1c, Elon., Up.Tri., Lo.Tri., In.Mid., Out.Mid.
        input = {}
        for param in input_params:
            input[param] = deepcopy(self.inputs[param])
        return input

    def simulate(self, actions):
        """Run the simulation for a specified number of steps."""
        seconds = 12
        idx = 0
        outputs = []
        controls = []
        inputs = [self.return_inputs()]
        
        # Predict 0D parameters first step 
        
        self.predict_0d(steady=True)
        self.first = False
        
        # βn, βp, h89, h98, q95, q0, li, wmhd
        outputs.append([v for v in self.return_outputs().values()])
        controls.append([v for v in self.return_controls().values()])

        # Mimic the behavior of the original code (they need one step before to update the visualizations)
        # Ip, Pnb1a, Pnb1b, Pnb1c, Elon., Up.Tri., Lo.Tri., In.Mid., Out.Mid.
        self.control(actions[idx])
        inputs.append(self.return_inputs())
        
        idx += 1
        self.predict_0d(steady=steady_model)
        outputs.append([v for v in self.return_outputs().values()])
        controls.append([v for v in self.return_controls().values()])
        
        # Main loop
        # p_bar = tqdm(total=seconds * 10)
        for sec in range(seconds):
            for step in range(10 - 1):
                
                self.control(actions[idx])
                inputs.append(self.return_inputs())
                
                idx += 1 
                self.predict_0d(steady=steady_model)
                outputs.append([v for v in self.return_outputs().values()])
                controls.append([v for v in self.return_controls().values()])
                
                # p_bar.update(1)
            
            self.control(actions[idx])
            inputs.append(self.return_inputs())
            
            idx += 1
            self.predict_0d(steady=steady_model)
            outputs.append([v for v in self.return_outputs().values()])
            controls.append([v for v in self.return_controls().values()])
            
            # p_bar.update(1)
            
        pkl.dump(inputs, open(os.path.join(self.vis_data_dir, 'inputs.pkl'), 'wb'))
        pkl.dump(outputs, open(os.path.join(self.vis_data_dir, 'outputs.pkl'), 'wb'))
        pkl.dump(controls, open(os.path.join(self.vis_data_dir, 'controls.pkl'), 'wb'))

        # return np.array(outputs)

    
# =======================
# Main Execution
# =======================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KSTAR Solver')
    # parser.add_argument('--id', type=int, default=0, help='Sample ID')
    parser.add_argument('--data_path', type=str, default='/public/dengwenhao/tokamak_data/', 
                        help='Path to the data')
    args = parser.parse_args()

    for i in range(10):
        data = np.load(os.path.join(args.data_path, f"{i}.npz"), allow_pickle=True)["data"].item()
        actions = data['actions'] # 121, 9
        # import pdb; pdb.set_trace()
        solver = KSTARSolver(sample_id=i)
        solver.simulate(actions) # 122. 8
        # print(f'Sample {i}:', np.linalg.norm(outputs - data['outputs']))

