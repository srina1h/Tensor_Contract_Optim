import torch
import torch.nn as nn
from datasets import load_dataset
import argparse
import time

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from tensor_layers.utils import config_class
from tensor_layers.Transformer_tensor import Transformer_classification

from transformers import BertTokenizer


def main():
    device = 'cuda'

    transformer = init_Transformer(num_class=3)

    transformer.load_state_dict(torch.load("model_real_data.pt"))
    
    batch_size = 1024
    training_data,validation_data = prepare_MNLI(batch_size)
    
    lr = 1e-3
    optimizer = optim.AdamW(transformer.parameters(),betas=(0.9, 0.999), eps=1e-09, lr = lr,weight_decay=0)
    
    config_forward = None

    eval_epoch(transformer, validation_data, device,config_forward=config_forward)


############# Prepare Dataset ################################

def prepare_MNLI(batch_size):
    def truncate(s1,s2,l=128):
        while len(s1)+len(s2)>l-3:
            if len(s1)>=len(s2):
                s1.pop()
            else:
                s2.pop()
        return s1,s2

    tokenize = BertTokenizer.from_pretrained("bert-base-uncased")
    def collate_fn_MNLI(batch):
        src_batch, tgt_batch,sim_batch = [], [],[]
        src_attn_batch = []
        seg_batch = []
        for c in batch:
        # for similar, src_sample, tgt_sample in batch:
            similar, src_sample, tgt_sample = c['label'],c['premise'],c['hypothesis']
            s1 = tokenize(src_sample)['input_ids'][1:-1]
            s2 = tokenize(tgt_sample)['input_ids'][1:-1]
            l = 128
            s1,s2 = truncate(s1,s2,l=l)

            tmp = [101] + s1 + [102] + s2 + [102]
            tmp = torch.tensor(tmp)
            
            seg = [0] + [0]*len(s1) + [0] + [1]*len(s2) + [1]

            src_batch.append(tmp)
            src_attn_batch.append(torch.tensor([1]*len(src_batch[-1])))
            sim_batch.append(similar)
            seg_batch.append(torch.tensor(seg))

        src_batch = pad_sequence(src_batch, padding_value=0)
        src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
        seg_batch = pad_sequence(seg_batch, padding_value=0)

        return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)


    train_iter = load_dataset("glue",'mnli',split='train')
    training_data = DataLoader(train_iter, batch_size=batch_size, collate_fn=collate_fn_MNLI, shuffle=True)

    val_iter = load_dataset("glue",'mnli',split='validation_matched')
    validation_data = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn_MNLI, shuffle=False)
    
    return training_data, validation_data


def init_Transformer(device='cuda',n_layers=12,num_class=2,embedding_rank=30,attention_rank=20,feedforward_rank=20,classification_rank=20):
    D = {
    'n_layers': n_layers,
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
    emb_rank = embedding_rank

    
    attn_shape = [12,8,8,8,8,12]
    pff_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
    attn_rank = attention_rank
    pff_rank = [feedforward_rank,feedforward_rank]

    classification_shape = [12,8,8,8,8,12]
    classification_rank = classification_rank


    config_model =config_class(**D)

    config_model.pff[0] = config_class(shape=pff_shape[0],ranks=pff_rank[0],set_scale_factors=set_scale_factors)
    config_model.pff[1] = config_class(shape=pff_shape[1],ranks=pff_rank[1],set_scale_factors=set_scale_factors)


    config_attn_sublayers = config_class(shape=attn_shape,ranks=attn_rank,set_scale_factors=set_scale_factors)
    for key in ['q','k','v','fc']:
        config_model.attn[key] = config_attn_sublayers


    config_model.embedding = config_class(shape=emb_shape,ranks=emb_rank,set_scale_factors=set_scale_factors)


    config_classification = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=num_class,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors)



    model = Transformer_classification(config_model,config_classification).to(device)
    
    return model



def train_epoch(model, training_data, optimizer,device='cuda',config_forward=None,optimizer_tensor=None):
    ''' Epoch operation in training phase'''
    

    model.train()




    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    cos_total = 0
    attn_total = 0

    count = 0

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        target, w1,attn,seg= map(lambda x: x.to(device), batch)
        

        optimizer.zero_grad()
        if optimizer_tensor!=None:
            optimizer_tensor.zero_grad()

        pred = model(w1,mask=attn,seg=seg,config_forward=config_forward)
        loss = Loss(pred,target)  
        
        

        loss.backward()

        
        
        optimizer.step()
        
        if optimizer_tensor!=None:
            optimizer_tensor.step()



        total_loss += loss.detach()
        n_word_total += pred.shape[0]
        n_word_correct += torch.sum(torch.argmax(pred.detach(),dim=1)==target)
        
        count+=1
        # if count==1:
        #     break


     
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy




def eval_epoch(model, validation_data, device, config_forward=None):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    Loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            target,w1,attn,seg = map(lambda x: x.to(device), batch)
      

            # pred = model(w1,mask=attn,seg=seg,config_forward=config_forward)
            # loss = Loss(pred,target)
            st = time.time()
            for i in range(50):
                with torch.no_grad():
                    pred = model(w1,mask=attn,seg=seg,config_forward=config_forward)
            ed = time.time()
            print(str((ed-st)/50)+"s inf time")
            loss = Loss(pred,target)
        
            total_loss += loss.item()


       
            n_word_total += pred.shape[0]
            n_word_correct += torch.sum(torch.argmax(pred,dim=1)==target)
            break

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy 
    
if __name__ == '__main__':
    main()