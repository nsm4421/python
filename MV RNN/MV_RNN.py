##---- Load modules ----##
import torch.nn as nn

##---- Multi Variable RNN----##
class MV_RNN(nn.Module):
    def __init__(self, model_name = 'RNN',
                observe_days = 7,    # Through observing <OBSERVE_DAY> days
                forcast_day = 3,     # Focast next <FORCAST_DAY> days
                input_features = 6, # Number of features of input data
                hidden_size = 5,
                num_layers = 1,     # Number of RNN layers
                drop_rate = 0
               ):
        super(MV_RNN, self).__init__()           

        #--- arguments ----#    
        self.model_name = model_name
        self.observe_days = observe_days
        self.forcast_day = forcast_day
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_rate = drop_rate
               
       
        #--- model ----#        
        try:     
            self.model_name = model_name 
            self.model = getattr(nn, model_name)(input_size = self.input_features,
                                                hidden_size = self.hidden_size,
                                                num_layers = self.num_layers,
                                                batch_first = True,
                                                dropout = self.drop_rate)
        except:
            self.model_name = 'RNN'           
            self.model = nn.RNN(input_size = self.input_features,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_layers,
                                batch_first = True,
                                dropout = self.drop_rate)
        

    def forward(self, x):
        output, _ = self.model(x)
        # output shape : [batch size, oberserve days, number of input features]
        output = output[:, :self.forcast_day, :]
        # output shape : [batch size, forcast days, number of output features]
        return output

    def help(self):
        print("model name : ['RNN', 'LSTM', 'GRU'] 중 하나 고름 됨 / default :  'RNN'")
        print("observe_days : 몇 일을 관찰?")
        print("forcast_day : 몇 일 뒤를 예측?")
        print("input_features : X의 feature 개수")
        print("output_features : Y의 feature 개수")
        print("hidden_size : RNN 모델에서 hidden size")
        print("num_layers : RNN layer 개수")      
        print("forcast_days보다 observe_days 이상으로 설정 ㄱㄱ")

    def shape(self):
        print('Input shape : [batch size, input sequence length, number of input features]')
        print(f'-----> batch size x {self.observe_days} x {self.input_features}')
        print('Output shape : [batch size, input sequence length, hidden size]')
        print(f'-----> batch size x {self.observe_days} x {self.hidden_size}')
  
        print('hidden state or cell state shape : [num layers, batch size, hidden size]')
        print(f'----->  {self.num_layers} x batch size x {self.hidden_size}')


        
