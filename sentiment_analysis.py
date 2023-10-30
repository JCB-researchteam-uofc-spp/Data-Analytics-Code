###################################################
##### Date: February 04,2021
##### Revision: V1.3
##### File : sentiment_analysis.py
##### Property of University of Calgary, Canada
###################################################
###################################################
###### Import Statements
###################################################
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import seaborn as sns
from collections import defaultdict
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import re
import string
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup


# Select CPU/GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

'''
Load the BERT tokenizer.
'''
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


'''
Dataloader 
For Training: Returns (Tweet, Input_id, Attention_mask, label)
For Testing: Returns (Tweet, Input_id, Attention_mask)
'''
class mydataset():    

    def __init__(self, classification_df, name = 'train'):

        super(mydataset).__init__()
        self.name = name
        self.tweet = []
        self.Y = []
                    
        for index,rows in classification_df.iterrows():

            tweet = rows['full_text']
            self.tweet.append(''.join(tweet)) 
            
            if name == 'train' or self.name == 'valid':
                label = rows['target']
                self.Y.append(label)

        
        
        '''
        Tokenize all of the captions and map the tokens to thier word IDs, and get respective attention masks.
        '''
        self.input_ids, self.attention_masks = tokenize(self.tweet)
        
        
    
    def __getitem__(self,index): 
        '''
        For Captions, Input ids and Attention mask
        '''
        tweet = self.tweet[index]
        input_id = self.input_ids[index]
        attention_masks = self.attention_masks[index]
        
        
        '''
        For Labels during training
        '''      
        if self.name == 'train' or self.name == 'valid' :
            label = float(self.Y[index])
            
            return tweet, input_id, attention_masks, torch.as_tensor(label).long()

        
        else:
            return tweet, input_id, attention_masks
        
        
  
    def __len__(self):
        return len(self.tweet)
    
    
            
'''
tokenize all of the sentences and map the tokens to their word IDs.
'''

def tokenize(sequences):
    
    input_ids = []
    attention_masks = []

    # For every caption...
    for seq in sequences:
        
        encoded_dict = tokenizer.encode_plus(
                            seq,                       # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 32,           # Pad & truncate all sentences.
                            truncation=True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',      # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    
    
    return input_ids, attention_masks

'''
Shuffle and split 10 percent data for Validation set
'''
train_csv = pd.read_csv('Sentiment_Analysis_train.csv', encoding='cp1252', keep_default_na = False)

def remove_tracks(text):
    text = str.lower(text)
    text = re.sub('(rt @[a-z0-9]+)\w+','', text)
    text = re.sub('(http\S+)', '', text)
    text = re.sub('([^0-9a-z \t])','', text)
    return text
train_csv['text'] = train_csv['full_text'].apply(lambda x: remove_tracks(x))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
train_csv['text'] = train_csv['text'].apply(lambda x: remove_punct(x))

ninety_percent  = round(0.90*(len(train_csv)))
train_data = train_csv.iloc[:ninety_percent]
valid_data = train_csv.iloc[ninety_percent:]

print('Number of Training samples: ', len(train_data))
print('Number of Validation samples: ',len(valid_data))

'''
Train Dataloader
''' 
train_dataset = mydataset(train_data, name = 'train')
train_dataloader = data.DataLoader(train_dataset, shuffle= True, batch_size = 32, num_workers=16,pin_memory=True)


'''
Validation_Dataloader
'''
validation_dataset = mydataset(valid_data, name = 'valid')
validation_dataloader = data.DataLoader(validation_dataset, shuffle= True, batch_size = 32, num_workers=16,pin_memory=True)

'''
Test Dataloader
''' 
test_csv = pd.read_csv('XXX.csv', encoding='utf-8', keep_default_na = False)
test_csv = test_csv.filter(['full_text'], axis=1)
test_dataset = mydataset(test_csv , name = 'test')          
test_dataloader = data.DataLoader(test_dataset, shuffle= False, batch_size = 1, num_workers=16,pin_memory=True)

'''
Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top.
''' 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", #12-layer BERT model, with an uncased vocab.
    num_labels = 3, #Number of Classes
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

model.to(device)

def train(model, data_loader, valid_loader, criterion, optimizer, lr_scheduler, modelpath, device, epochs):
    
    model.train()

    train_loss= []
    valid_loss = []
    valid_acc = []


    for epoch in range(epochs):
        avg_loss = 0.0
                
        
        for batch_num, (tweet, input_id, attention_masks, target) in enumerate(data_loader):
            
            input_ids, attention_masks, target = input_id.to(device), attention_masks.to(device), target.to(device)
                
            '''
            Compute output and loss from BERT
            '''
            loss, logits = model(input_ids, 
                             token_type_ids=None, 
                             attention_mask=attention_masks, 
                             labels=target,
                             return_dict=False
                                )

            '''
            Take Step
            '''                    
            optimizer.zero_grad()
            loss.backward()
            # order changes as per suggestion by IV.ai
            '''
            Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
            '''
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            
            optimizer.step()


            

            
            avg_loss += loss.item()

            '''
            linear_schedule_with_warmup take step after each batch
            '''
            lr_scheduler.step()
                                
            
        training_loss = avg_loss/len(data_loader)
       
        print('Epoch: ', epoch+1)            
        print('training loss = ', training_loss)
        train_loss.append(training_loss)

        '''
        Check performance on validation set after an Epoch
        '''
        validation_loss, top1_acc= test_classify(model, valid_loader, criterion, device)
        valid_loss.append(validation_loss)
        valid_acc.append(top1_acc)

         
        '''
        save model checkpoint after every epoch
        '''
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            }, modelpath)
        
    return train_loss, valid_loss, valid_acc






'''
Function to perform inference on validation set
Returns: validation loss, top1 accuracy
'''

def test_classify(model, valid_loader, criterion, device):
    model.eval()
    test_loss = []
    top1_accuracy = 0
    total = 0

    for batch_num, (tweet, input_id, attention_masks, target) in enumerate(valid_loader):
               
        input_ids, attention_masks, target = input_id.to(device), attention_masks.to(device), target.to(device)
            
        '''
        Compute output and loss from BERT
        '''
        loss, logits = model(input_ids, 
                         token_type_ids=None, 
                         attention_mask=attention_masks, 
                         labels=target,
                         return_dict=False)

        test_loss.extend([loss.item()]*input_id.size()[0])
        
        predictions = F.softmax(logits, dim=1)
        
        _, top1_pred_labels = torch.max(predictions,1)
        top1_pred_labels = top1_pred_labels.view(-1)
        
        top1_accuracy += torch.sum(torch.eq(top1_pred_labels, target)).item()
        total += len(target)

    print('Validation Loss: {:.4f}\tTop 1 Validation Accuracy: {:.4f}'.format(np.mean(test_loss), top1_accuracy/total))
        
    return np.mean(test_loss), top1_accuracy/total


'''
Loss Function
'''
criterion = nn.CrossEntropyLoss()

'''
Optimizer
'''
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

'''
Number of training epochs. The BERT authors recommend between 2 and 4. Increasing the number of epochs with BERT will increase overfitting the training set, as it can be seen from the loss plot later.
'''
num_Epochs = 3

'''
Create the learning rate scheduler.
Total number of training steps is [number of batches] x [number of epochs].
'''
total_steps = len(train_dataloader) * num_Epochs
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,  num_training_steps = total_steps)

modelname = 'BERT'
modelpath = 'saved_checkpoint_'+modelname

train_loss, valid_loss, valid_acc = train(model, train_dataloader, validation_dataloader, criterion, optimizer, lr_scheduler, modelpath, device, epochs = num_Epochs)

def plot_loss(epochs, train_loss, test_loss, title):
    plt.figure(figsize=(8,8))
    matplotlib.use('Agg')
    plt.ioff()
    x = np.arange(1,epochs+1)
    plt.plot(x, train_loss, label = 'Training Loss')
    plt.plot(x, test_loss, label = 'Validation Loss')
    plt.xlabel('Epochs', fontsize =16)
    plt.ylabel('Loss', fontsize =16)
    plt.title(title,fontsize =16)
    plt.legend(fontsize=16)
    plt.savefig('Loss plot_new.png', dpi = 100)
    
def plot_acc(epochs,test_acc):
    plt.figure(figsize=(8,8))
    matplotlib.use('Agg')
    plt.ioff()
    x = np.arange(1,epochs+1)
    plt.plot(x, test_acc)
    plt.xlabel('Epochs', fontsize =16)
    plt.ylabel('Test Accuracy', fontsize =16)
    plt.title('Test Accuracy v/s Epochs',fontsize =16)
    plt.savefig('Test Accuracy.png', dpi = 100)
    
#sns.set_style("whitegrid")
matplotlib.use('Agg')
plt.ioff()
plot_loss(num_Epochs, train_loss, valid_loss, title='Loss plot')
plot_acc(num_Epochs, valid_acc)

def predict(model, test_loader, device):
    model.eval()
    target = []
    for batch_num, (captions, input_id, attention_masks) in enumerate(test_loader):
     
        
        input_ids, attention_masks = input_id.to(device), attention_masks.to(device)
            
        '''
        Compute prediction outputs from BERT
        '''
        output_dictionary = model(input_ids, 
                         token_type_ids=None, 
                         attention_mask=attention_masks, 
                         return_dict=True)
        
        predictions = F.softmax(output_dictionary['logits'], dim=1)
        
        _, top1_pred_labels = torch.max(predictions,1)
        top1_pred_labels = top1_pred_labels.item()        
        target.append(top1_pred_labels)
        
        
    make_csv(target)
        

def make_csv(target):
    test = pd.read_csv('XXX.csv', encoding='utf-8', keep_default_na = False)
    test = test.rename(columns={"Unnamed: 0" : "id"})
    my_submission = pd.DataFrame({'id': test.id, 'target': target})
    my_submission.to_csv('XXX.csv', index=False)
    

predict(model, test_dataloader, device)

