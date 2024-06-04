import sys
sys.path.append("../")
from utils.experiment_utils import mean_dfs
import numpy as np
import pandas as pd
from utils.utils_base import *

from itertools import *
from utils.utils_base import *
from utils.data_loader import *
from method.diffnaps import *
from utils.data_loader import  perm
from utils.gen_synth_datasets import *
from method.my_loss import weightedXorCover

from utils.measures import *
import time, datetime
from torch.utils.tensorboard import SummaryWriter

def cleanup(folder_path, zip_path):
      # zip result
      import zipfile
      import shutil

      def zip_folder(folder_path, zip_path):
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                  for root, dirs, files in os.walk(folder_path):
                        for file in files:
                              file_path = os.path.join(root, file)
                              zipf.write(file_path, os.path.relpath(file_path, folder_path))

      # Example usage
      #zip_path = '/path/to/archive.zip'
      zip_folder(folder_path, zip_path)
      # del result
      print(f"del {folder_path}")
      shutil.rmtree(folder_path)

def gen_configs():
      conf = dict()
      conf['diff'] = TrainConfig(hidden_dim = 1000, epochs=50, weight_decay=2, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                               lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=32, wd_class=0.0,
                         log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="", save_xp=True,)
     
      conf['bc'] = TrainConfig(hidden_dim = 2000, epochs=100, weight_decay=2, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                               lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=32, wd_class=0.0,
                         log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                              loss=weightedXorCover, alpha=10,save_xp=True,
                         )
    
      conf['binaps'] = TrainConfig(hidden_dim = 1000, epochs=50, weight_decay=2, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                               lambda_c = 10, regu_rate=1.08, class_regu_rate=1.08, batch_size=32, wd_class=0.0,
                         log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,init_enc="",
                              loss=weightedXorCover, alpha=2000,save_xp=True,
                         )

      return conf

def exp3(output_dir, **kwargs):
      if not torch.cuda.is_available():
            device = torch.device("cpu")
            print("WARNING: Running purely on CPU. Slow.")
      else:
            device = torch.device("cuda")
      """Test bruit additif """
      dfs = []
      snrs = [0,10,20,30,40,50,60,70,80,90,100]
      # snrs = [50]
      # snrs = [0]
      # columns = ['method', "ncols","JaccarD","SoftPrec","SoftRecall","F1","cov", "pat_count", "purity", "redon", "supp", "wf1(quant)", "time"]
      root = os.path.join(os.path.dirname(os.path.realpath(__file__)), r"../results/synth_results/diffnaps/")

      for seed in [0,1,2,3,4]:
      # for seed in [0]:
            conf_exp3_all = gen_configs()
            # avg_res_list = []
            df=pd.DataFrame()

            for k in snrs:
                  exp_dict = experiment3([k],seed=seed)

                  for method, conf_exp3 in conf_exp3_all.items():
                        log_dir = os.path.join(root, r"runs/")
                        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        tag =  f'{current_time}_exp3_{method}_k{k}_s{seed}_a{conf_exp3.alpha}'
                        run_dir = log_dir +  tag 
                        writer = SummaryWriter(log_dir=run_dir)
                        writer.add_text('Run info', 'Hyperparameters:' + conf_exp3.to_str())

                        print(10*"#",k,seed,10*"#")

                        data, labels, gt_dict = exp_dict[k]
                        data, labels = perm(data.astype(np.float32),labels.astype(int))
                        gt_dict = {str(k):v for k,v in gt_dict.items()}
                        start_time = time.time()
                        # Train diffnaps
                        # model, new_weights, trainDS, test_loader = learn_diffnaps_net(data, conf_exp3, labels = labels,ret_test=True,  writer=writer, verbose=True)
                        
                        if method == "binaps":
                              model, new_weights, trainDS, test_loader = learn_xp_binaps(data,conf_exp3, labels = labels,ret_test=True, writer=writer, verbose=False)
                        else:
                              model, new_weights, trainDS, test_loader = learn_diffnaps_net(data,conf_exp3, labels = labels,ret_test=True,verbose=False, writer=writer)
                       
                        
                        time_taken = time.time() - start_time
                        time_taken = time_taken / 60

                        enc_w = model.fc0_enc.weight.data.detach().cpu()
                        print("Encoder mean: ",enc_w.mean())
                        print("Encoder: ",enc_w.std())
                        if conf_exp3.save_xp:
                              file = os.path.join(run_dir, "model_weight.pt")
                              torch.save(enc_w, file)
                              file = os.path.join(run_dir, "data")
                              np.save(file,data)
                              file = os.path.join(run_dir, "labels")
                              np.save(file, labels)
                              file = os.path.join(run_dir, 'ground_truth.json')
                              with open(file, 'w') as fd:
                                    json.dump(gt_dict, fd)
                        # extract the differntial patterns, t1 is t_e and t2 is t_c 
                        # _,_,_,num_pat,patterns, gen_patterns = get_positional_patterns(enc_w,c_w, general=True, t_mean=1,  t1=enc_bin,t2=class_bin, device=device)
                        if method == 'binaps':
                              _,_,_,num_pat,patterns, gen_patterns = get_positional_patterns_binaps(weights=enc_w,data=data, labels=labels, general=True)

                        else:
                              c_w = model.classifier.weight.detach().cpu()
                              if conf_exp3.save_xp:
                                    file = os.path.join(run_dir, "classif_weight.pt")
                                    torch.save(c_w, file)
                              _,_,_,num_pat,patterns, gen_patterns = get_positional_patterns(enc_w,c_w, general=True, t_mean=1,  t1=0.3,t2=0.3, device=device)
                       
                        metric_result = {'method':method,'Anoise':k, 'ktop':0, 'NMF':'_'}
                        metric_result.update(mean_overlap_function(patterns, gt_dict))
                        metric_result.update(mean_compute_scores(patterns,gt_dict))
                        metric_result.update(mean_compute_metric(data, labels, patterns, device=device))
                  
                        for key, value in metric_result.items():
                              if key != "method" and key != "NMF":
                                    writer.add_scalar(key, value, k)
                        df = pd.concat([df, pd.DataFrame(metric_result, index=[0])], ignore_index=True)

                        if method == 'bc':
                              new_p = compile_new_pat_by_class(labels=labels, patterns=patterns, data=data, n=[2,3], device=device, max_iter=1000, rank=10)

                              for keyk, val in new_p.items():

                                    metric_result = {'method':method,'Anoise':k, 'ktop':keyk, 'NMF':'filter'}
                                    metric_result.update(mean_overlap_function(val, gt_dict))
                                    metric_result.update(mean_compute_scores(val,gt_dict))
                                    metric_result.update(mean_compute_metric(data, labels, val, device=device))
                                    df = pd.concat([df, pd.DataFrame(metric_result, index=[0])], ignore_index=True)
                                    for key, value in metric_result.items():
                                          if key != "method" and key != "NMF":
                                                writer.add_scalar(key+f'/filter/{keyk}', value, k)

                              # new_p = compile_new_pat_whole(labels=labels, patterns=patterns, data=data, n=[2,3], device=device, max_iter=2000, rank=50)

                              # for keyk, val in new_p.items():

                              #       metric_result = {'method':method, 'Anoise':k,'ktop':keyk, 'NMF':'whole'}
                              #       metric_result.update(mean_overlap_function(val, gt_dict))
                              #       metric_result.update(mean_compute_scores(val,gt_dict))
                              #       metric_result.update(mean_compute_metric(data, labels, val, device=device))
                              #       df = pd.concat([df, pd.DataFrame(metric_result, index=[0])], ignore_index=True)
                              #       for key, value in metric_result.items():
                              #             if key != "method" and key != "NMF":
                              #                   writer.add_scalar(key+f'/whole/{keyk}', value, k)


                        del model
                        writer.close()
                        
                        if output_dir:
                              cleanup(run_dir,os.path.join(output_dir, tag+'.zip'))

            file = r"exp3_diffnaps_%d.xlsx"%seed
            # file = r"../results/synth_results/diffnaps/exp3_diffnaps_%d.csv"%seed
            if output_dir:
                  df.to_excel(os.path.join(output_dir, file),index=False)
            else:
                  df.to_excel(os.path.join(root, file),index=False)
            dfs.append(df)
      mean_dfs(dfs, "diffnaps", "exp3_diffnaps", output_dir)

      
if __name__ == "__main__":
      import argparse
      parser = argparse.ArgumentParser(description='Process some input argument.')
      parser.add_argument('--output', type=str, help='copy output to this dir', default=None)
      args = parser.parse_args()

      # output = r"C:\Users\Thibaut\OneDrive - Palo IT\Documents\These\code\test_ju\diffnaps\results\synth_results\diffnaps\runs"
      exp3(args.output)