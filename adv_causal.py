print('this file is adv_causal, the time is: ')


import datetime
now = datetime.datetime.now()
bj_time = now + datetime.timedelta(hours=8)
print(bj_time.strftime('%Y-%m-%d %H:%M:%S'))

import argparse
import os
import time

import torch

from beer import BeerData, BeerAnnotation,Beer_correlated
from hotel import HotelData,HotelAnnotation
from embedding import get_embeddings,get_glove_embedding
from torch.utils.data import DataLoader

from model import Sp_norm_model_causal
from train_util import train_adv_causal, dev_adv_causal
from validate_util import validate_share, validate_dev_sentence, validate_annotation_sentence, validate_rationales
from tensorboardX import SummaryWriter


def parse():
    #默认： nonorm, dis_lr=1, data=beer, save=0
    parser = argparse.ArgumentParser(
        description="SR")

    parser.add_argument('--lr_lambda',
                        type=int,
                        default=1,
                        help='')


    # pretrained embeddings
    parser.add_argument('--embedding_dir',
                        type=str,
                        default='./data/hotel/embeddings',
                        help='Dir. of pretrained embeddings [default: None]')
    parser.add_argument('--embedding_name',
                        type=str,
                        default='glove.6B.100d.txt',
                        help='File name of pretrained embeddings [default: None]')
    parser.add_argument('--max_length',
                        type=int,
                        default=256,
                        help='Max sequence length [default: 256]')

    # dataset parameters
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/beer',
                        help='Path of the dataset')
    parser.add_argument('--data_type',
                        type=str,
                        default='beer',
                        help='0:beer,1:hotel')
    parser.add_argument('--correlated',
                        type=int,
                        default=1,
                        help='The aspect number of beer review [0, 1]')
    parser.add_argument('--aspect',
                        type=int,
                        default=0,
                        help='The aspect number of beer review [0, 1, 2]')
    parser.add_argument('--seed',
                        type=int,
                        default=12252018,
                        help='The aspect number of beer review [20226666,12252018]')
    parser.add_argument('--annotation_path',
                        type=str,
                        default='./data/beer/annotations.json',
                        help='Path to the annotation')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size [default: 100]')


    # model parameters
    parser.add_argument('--save',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--gen_acc',
                        type=int,
                        default=0,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--gen_sparse',
                        type=int,
                        default=1,
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--div',
                        type=str,
                        default='kl',
                        help='save model, 0:do not save, 1:save')
    parser.add_argument('--cell_type',
                        type=str,
                        default="GRU",
                        help='Cell type: LSTM, GRU [default: GRU]')
    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='RNN cell layers')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2,
                        help='Network Dropout')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=100,
                        help='Embedding dims [default: 100]')
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='RNN hidden dims [default: 100]')
    parser.add_argument('--num_class',
                        type=int,
                        default=2,
                        help='Number of predicted classes [default: 2]')
    parser.add_argument('--lay',
                        type=bool,
                        default=True,
                        help='Number of predicted classes [default: 2]')
    parser.add_argument('--model_type',
                        type=str,
                        default='sp',
                        help='Number of predicted classes [default: 2]')

    # ckpt parameters
    parser.add_argument('--output_dir',
                        type=str,
                        default='./res',
                        help='Base dir of output files')

    # learning parameters
    parser.add_argument('--dis_lr',
                        type=int,
                        default=0,
                        help='Number of training epoch')
    parser.add_argument('--epochs',
                        type=int,
                        default=37,
                        help='Number of training epoch')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='compliment learning rate [default: 1e-3]')
    parser.add_argument('--sparsity_lambda',
                        type=float,
                        default=12.,
                        help='Sparsity trade-off [default: 1.]')
    parser.add_argument('--continuity_lambda',
                        type=float,
                        default=10.,
                        help='Continuity trade-off [default: 4.]')
    parser.add_argument(
        '--sparsity_percentage',
        type=float,
        default=0.1,
        help='Regularizer to control highlight percentage [default: .2]')
    parser.add_argument(
        '--cls_lambda',
        type=float,
        default=0.9,
        help='lambda for classification loss')
    parser.add_argument(
        '--div_lambda',
        type=float,
        default=1,
        help='lambda for js divergence')
    parser.add_argument(
        '--x_lambda',
        type=float,
        default=1,
        help='lambda for full text')
    parser.add_argument('--gpu',
                        type=str,
                        default='3',
                        help='id(s) for CUDA_VISIBLE_DEVICES [default: None]')
    parser.add_argument('--share',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument(
        '--writer',
        type=str,
        default='./noname',
        help='Regularizer to control highlight percentage [default: .2]')
    args = parser.parse_args()
    return args


#####################
# set random seed
#####################
# torch.manual_seed(args.seed)

#####################
# parse arguments
#####################
args = parse()
torch.manual_seed(args.seed)
print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

######################
# device
######################
torch.cuda.set_device(int(args.gpu))
device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(args.seed)

######################
# load embedding
######################
pretrained_embedding, word2idx = get_glove_embedding(os.path.join(args.embedding_dir, args.embedding_name))
args.vocab_size = len(word2idx)
args.pretrained_embedding = pretrained_embedding

######################
# load dataset
######################
if args.data_type=='beer':       #beer
    if args.correlated==0:
        print('decorrelated')
        train_data = BeerData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

        dev_data = BeerData(args.data_dir, args.aspect, 'dev', word2idx)
    else:
        print('correlated')
        train_data = Beer_correlated(args.data_dir, args.aspect, 'train', word2idx, balance=True)

        dev_data = Beer_correlated(args.data_dir, args.aspect, 'dev', word2idx,balance=True)

    annotation_data = BeerAnnotation(args.annotation_path, args.aspect, word2idx)
elif args.data_type == 'hotel':       #hotel
    args.data_dir='./data/hotel'
    args.annotation_path='./data/hotel/annotations'
    train_data = HotelData(args.data_dir, args.aspect, 'train', word2idx, balance=True)

    dev_data = HotelData(args.data_dir, args.aspect, 'dev', word2idx)

    annotation_data = HotelAnnotation(args.annotation_path, args.aspect, word2idx)

# shuffle and batch the dataset
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

annotation_loader = DataLoader(annotation_data, batch_size=args.batch_size)

######################
# load model
######################
writer=SummaryWriter(args.writer)
model=Sp_norm_model_causal(args)
model.to(device)

######################
# Training
######################

p_para=[]
for p in model.cls.parameters():
    if p.requires_grad==True:
        p_para.append(p)
for p in model.cls_fc.parameters():
    if p.requires_grad==True:
        p_para.append(p)


lr2=args.lr/args.lr_lambda
lr1=args.lr

g_para=filter(lambda p: p.requires_grad==True, model.generator.parameters())
para_gen=[{'params': g_para, 'lr':lr1}]
para_pred=[{'params':p_para,'lr':lr2}]


optimizer_gen = torch.optim.Adam(para_gen)
optimizer_pred=torch.optim.Adam(para_pred)

# optimizer = torch.optim.Adam(model.parameters())

######################
# Training
######################
strat_time=time.time()
best_all = 0
f1_best_dev = [0]
best_dev_epoch = [0]
acc_best_dev = [0]
grad=[]
grad_loss=[]
for epoch in range(args.epochs):

    start = time.time()
    model.train()
    rationale_classify= train_adv_causal(model,optimizer_gen,optimizer_pred, train_loader, device, args,(writer,epoch))

    end = time.time()
    print('\nTrain time for epoch #%d : %f second' % (epoch, end - start))
    # print('gen_lr={}, pred_lr={}'.format(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
    print("rationale classification. recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(rationale_classify[0],
                                                                                                   rationale_classify[1], rationale_classify[2],
                                                                                                   rationale_classify[3]))




    writer.add_scalar('time',time.time()-strat_time,epoch)
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    model.eval()
    print("Validate")


    rationale_dev= dev_adv_causal(model, train_loader, device, args, (writer, epoch))

    print("rationale classification. recall:{:.4f} precision:{:.4f} f1-score:{:.4f} accuracy:{:.4f}".format(
        rationale_dev[0],
        rationale_dev[1], rationale_dev[2],
        rationale_dev[3]))




    print("Annotation")
    annotation_results = validate_share(model, annotation_loader, device)
    print(
        "The annotation performance: sparsity: %.4f, precision: %.4f, recall: %.4f, f1: %.4f"
        % (100 * annotation_results[0], 100 * annotation_results[1],
           100 * annotation_results[2], 100 * annotation_results[3]))
    writer.add_scalar('f1',100 * annotation_results[3],epoch)
    writer.add_scalar('sparsity',100 * annotation_results[0],epoch)
    writer.add_scalar('p', 100 * annotation_results[1], epoch)
    writer.add_scalar('r', 100 * annotation_results[2], epoch)


    if best_all<annotation_results[3]:
        best_all=annotation_results[3]
print(best_all)
print(acc_best_dev)
print(best_dev_epoch)
print(f1_best_dev)
if args.save==1:
    if args.data_type=='beer':
        torch.save(model.state_dict(),'./trained_model/beer/aspect{}_dis{}.pkl'.format(args.aspect,args.dis_lr))
        print('save the model')
    elif args.data_type=='hotel':
        torch.save(model.state_dict(), './trained_model/hotel/aspect{}_dis{}.pkl'.format(args.aspect, args.dis_lr))
        print('save the model')
else:
    print('not save')