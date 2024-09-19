import argparse
import torch
import os
import pandas as pd
import logging
import datetime
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from method.diffnaps import learn_diffnaps_net
from method.network import learn_xp_binaps
from method.dataLoader import readDatFile
from utils.utils_base import TrainConfig, get_positional_patterns, get_positional_patterns_binaps, mean_compute_metric, compile_new_pat_by_class
from utils.experiment_utils import roc, res_to_csv, write_time
    
NOW = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Create a custom log
log = logging.getLogger('root')
# log.propagate = False
# Set the log level
log.setLevel(logging.DEBUG)
# Create a console handler and set the log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')
ch.setFormatter(formatter)
# Add the handler to the log
log.addHandler(ch)


log.info('init')


def main(args=None):
    # Training settings
    parser = argparse.ArgumentParser(description='Diffversify & co implementation')
    parser.add_argument('-i','--input', required=True,
                        help='Input file to use for training and testing (.dat format)')
    parser.add_argument('-l','--label', required=False,
                        help='Input label to use for training and testing (.dat format)')
    parser.add_argument('-m','--method', required=True,
                        help='Which model to use : [diffversify, binaps, diffnaps]')
    parser.add_argument('-o','--output', default="",
                        help='Output directory')
    
    parser.add_argument('--train_set_size', type=float, default=.9,
                        help='proportion of data to be used for training')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--hidden_dim', type=int, default=-1,
                        help='size for the hidden layer (default: #features)')
    
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay for L2 norm (default 0)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--elb_k', type=float, default=1,
                        help='elb regulation (default: 1)')
    parser.add_argument('--elb_lamb', type=float, default=1,
                        help='elb regulation (default: 1)')
    parser.add_argument('--class_elb_k', type=float, default=1,
                        help='elb regulation (default: 1)')
    parser.add_argument('--class_elb_lamb', type=float, default=1,
                        help='elb regulation (default: 1)')
    parser.add_argument('--lambda_c', type=float, default=0.1,
                        help='Lambda to equilibrate loss (default: 0.1)')
    parser.add_argument('--regu_rate', type=float, default=1.08,
                        help='regulation rate for elb (default: 1.08)')
    parser.add_argument('--class_regu_rate', type=float, default=1.08,
                        help='regulation rate for elb (default: 1.08)')
    parser.add_argument('--sparse_regu', type=float, default=0,
                        help='Sparse regulation (default: 0)')
    parser.add_argument('--alpha', type=float, default=1,
                        help='Contrastvive divergence weight (default: 1)')
    # parser.add_argument('--k_w', type=float, default=0.1,
    #                     help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--k_f', type=float, default=100,
                        help='rank for NMF')
    parser.add_argument('--t1', type=float, default=0.01,
                        help='Threshold for binarization of the embedding')
    parser.add_argument('--t2', type=float, default=0.02,
                        help='Threshold for binarization of the classifier')
    
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='save the current Model')
    parser.add_argument('--save_xp', action='store_true', default=False,
                        help='save the current Model')
    parser.add_argument('--test', action='store_true', default=False,
                        help='save the current Model')
    parser.add_argument('--thread_num', type=int, default=16,
                        help='number of threads to use (default: 16)')
    
    args = parser.parse_args(args)
    output = os.path.join(args.output, NOW)

    config = TrainConfig().lazy_load_from_dict(args.__dict__)
    
    writer = SummaryWriter(output)
    writer.add_text('Run info', 'Hyperparameters:' + config.to_str())
    
    fl = logging.FileHandler(filename=os.path.join(output, f'{args.method}.log'))
    fl.setFormatter(formatter)
    fl.setLevel(logging.DEBUG)
    log.addHandler(fl)
    log.debug(f'ARGS : {args.__dict__}')
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        log.warning("WARNING: Running purely on CPU. Slow.")
    else:
        device = torch.device("cuda")
        log.info(f'Running on {device}')

    log.debug(config.__dict__)


    # load data
    log.info(f'Loading data : {args.input}')
    path_data = os.path.join(args.input)
    data = readDatFile(args.input)

    log.info(f'Loading labels : {args.label}')
    with open(args.label, 'r') as fd: 
       labels = fd.read().split('\n')
    
    labels = np.array(labels).reshape(-1)

    label_dict = {lab : i for i, lab in enumerate(np.unique(labels))}
    labels = np.array([label_dict[lab] for lab in labels])
    label_dict = {v: k for k, v in label_dict.items()}

    translator = [i for i in range(data.shape[1])]

    start_time = time.time()
    if args.method == "binaps":
            log.info("Training with BiNaps")
            model, new_weights, trainDS, test_loader = learn_xp_binaps(data, config, labels = labels,ret_test=True, writer=writer, verbose=True)
    else:
            log.info(f"Training with {args.method}")
            model, new_weights, trainDS, test_loader = learn_diffnaps_net(data, config, labels = labels,ret_test=True,verbose=True, writer=writer)
                    
    time_taken = time.time() - start_time
    time_taken = time_taken / 60
    enc_w = model.fc0_enc.weight.data.detach().cpu()
    
    if config.save_xp:
            log.debug(f"Save XP to {output}")
            file = os.path.join(output, "model_weight.pt")
            torch.save(enc_w, file)
            file = os.path.join(output, "data")
            np.save(file,data)
            file = os.path.join(output, "labels")
            np.save(file, labels)

    
    if args.method == 'binaps':
        log.info("Extracting pattern for BiNaps")
        _,_,_,num_pat,res_dict, gen_patterns = get_positional_patterns_binaps(weights=enc_w,data=data, labels=labels, general=True)
    else:
        log.info(f"Extracting pattern for {args.method}")
        c_w = model.classifier.weight.detach().cpu()
        if config.save_xp:
            file = os.path.join(output, "classif_weight.pt")
            torch.save(c_w, file)
        _,_,_,num_pat,res_dict, gen_patterns = get_positional_patterns(enc_w,c_w, general=True, t_mean=1, t1=config.t1,t2=config.t2, device=device)
                                                                      
                    
    res_dict_int = {int(k):v for k,v in res_dict.items()}
    res_to_csv(args.method, os.path.basename(args.input), res_dict_int, data, labels, label_dict, translator, output=output)

    log.info("Get metrics without NMF")
    metric_result = {'method':args.method, 'db':os.path.basename(args.input), 'ktop':0, 'NMF':'_'}
    metric_result.update(mean_compute_metric(data, labels, res_dict, device=device))
    if metric_result['pat_count']==0:
        logging.error('ALERT ! No pattern found !')
        return
    # line_x, line_y, auc = roc(res_dict_int, data,labels,label_dict,translator,verbose=False)
    # metric_result['roc_auc'] = auc
    
    df_metric = pd.DataFrame(metric_result, index=[0])

    for key, value in metric_result.items():
        if key != "method" and key != 'db' and key != 'NMF':
            writer.add_scalar(key, value,0)

    # dr = pd.DataFrame({'x':line_x, 'y':line_y})
    # dr.to_csv(os.path.join(output, 'auc_roc_data_noNMF.csv'))



    if args.method == 'diffversify':
        log.info("NMF for DiffVersify with /!\ HARDCODED TOP-k = 2")
        new_p = compile_new_pat_by_class(labels=labels, patterns=res_dict, data=data, n=[2], device=device, max_iter=500, rank=config.k_f)

        for keyk, val in new_p.items():

            metric_result = {'method':args.method,'db':os.path.basename(args.input), 'ktop':keyk, 'NMF':'filter'}
            metric_result.update(mean_compute_metric(data, labels, val, device=device))
            res_dict_int = {int(k):v for k,v in val.items()}
            # line_x, line_y, auc = roc(res_dict_int, data,labels,label_dict,translator,verbose=False)
            # metric_result['roc_auc'] = auc
            df_metric = pd.concat([df_metric, pd.DataFrame(metric_result, index=[0])], ignore_index=True)
            
            for key, value in metric_result.items():
                if key != "method" and key != 'db' and key != 'NMF':
                            writer.add_scalar(key, value, keyk)

            # dr = pd.DataFrame({'x':line_x, 'y':line_y})
            # dr.to_csv(os.path.join(output, f'auc_roc_data_NMF_filter_{keyk}.csv'))
            res_to_csv(args.method, os.path.basename(args.input)+'_NMF'+str(keyk), res_dict_int, data, labels, label_dict, translator, output=output)

        file = rf"{os.path.basename(args.input)}_{args.method}.xlsx"
        df_metric.to_excel(os.path.join(output, file),index=False)

    log.info(f'Time taken = {time_taken}')
    writer.close()





if __name__ == "__main__":
   pass
    # argu="-i C:\Users\chataint\Documents\projet\diffversify\runs\data.dat -o C:\Users\chataint\Documents\projet\diffversify\runs\out -m binaps"