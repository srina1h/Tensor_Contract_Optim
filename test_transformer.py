import torch
from tensor_layers.utils import config_class
from tensor_layers.Transformer_tensor import Transformer_classification
import time

device = 'cuda'

D = {
    'n_layers': 12,
    'vocab_size': 30522,
    'n_position': 512,
    'd_model':768,
    'd_hid':768*4,
    'n_head':12,
    'tensorized':True,
    'dropout': 0.1,
    'embedding': None,
    'classification': None,
    'pff': {},
    'attn': {}
    }

set_scale_factors = False

emb_shape = [[16,20,10,10],[4,4,8,6]]
emb_rank = 30

r = 20
attn_shape = [12,8,8,8,8,12]
pff_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
attn_rank = r
pff_rank = [r,r]

classification_shape = [12,8,8,8,8,12]
classification_rank = 20


config_model =config_class(**D)

config_model.pff[0] = config_class(shape=pff_shape[0],ranks=pff_rank[0],set_scale_factors=set_scale_factors)
config_model.pff[1] = config_class(shape=pff_shape[1],ranks=pff_rank[1],set_scale_factors=set_scale_factors)


config_attn_sublayers = config_class(shape=attn_shape,ranks=attn_rank,set_scale_factors=set_scale_factors)
for key in ['q','k','v','fc']:
    config_model.attn[key] = config_attn_sublayers


config_model.embedding = config_class(shape=emb_shape,ranks=emb_rank,set_scale_factors=set_scale_factors)


num_class = 2

config_classification = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=num_class,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors)



model = Transformer_classification(config_model,config_classification).to(device)
model.load_state_dict(torch.load("model.pt"))


input = torch.randint(0,30000,(32,128)).to(device)

model.eval()
st = time.time()
for i in range(50):
    with torch.no_grad():
        model(input)
ed = time.time()
print(str((ed-st)/50)+"s inf time")