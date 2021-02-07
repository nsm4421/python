import MV_RNN    # my module
import torch

BATCH_SIZE = 32
OBSERVE = 7
FORCAST = 3
N_INPUT_FEATURE = 6
N_OUTPUT_FEATURE = 5

x = torch.rand([BATCH_SIZE, OBSERVE, N_INPUT_FEATURE])
print(f'shape of x : {x.shape}')

rnn_model = MV_RNN.MV_RNN(model_name = 'RNN',
                    observe_days = OBSERVE,   
                    forcast_day = FORCAST,     
                    input_features = N_INPUT_FEATURE, 
                    hidden_size = N_OUTPUT_FEATURE)

y = rnn_model(x)
print(f'shape of y : {y.shape}')
