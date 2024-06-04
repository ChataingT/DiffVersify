import sys
sys.path.append("../")
from utils.experiment_utils import mean_dfs
from utils.utils_base import *
from itertools import *
from utils.utils_base import *
from utils.data_loader import *
from method.diffnaps import *
from utils.data_loader import  perm
from utils.gen_synth_datasets import *
from utils.experiment_utils import *
from utils.measures import *
import time, datetime
import argparse, random
from method.my_loss import weightedXorCover
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


def get_conf_diff():
    conf_dict = {}

    conf_dict["cardio"] = TrainConfig(hidden_dim = 500, epochs=25, weight_decay=8.0, elb_k=1, elb_lamb=1, class_elb_k=5, class_elb_lamb=5,
                                lambda_c = 100.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True)
    conf_dict["cardio"].t1 = 0.15
    conf_dict["cardio"].t2 = 0.1


    conf_dict["disease"] = TrainConfig(hidden_dim = 1500, epochs=80, weight_decay=9.0, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                lambda_c = 50, regu_rate=1.08, class_regu_rate=1.08, batch_size=32,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True, k_w=300, k_f=100)
    conf_dict["disease"].t1=0.15
    conf_dict["disease"].t2=0.1



    conf_dict["brca-n"] = TrainConfig(hidden_dim = 10000, epochs=20, weight_decay=5.0,vertical_decay=0.0, elb_k=1, elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                lambda_c = 10.0, regu_rate=1.08, class_regu_rate=1.08, batch_size = 32,
                            log_interval=100, sparse_regu=0,init_enc="", test=False, lr=0.005, seed=0, model=DiffnapsNet,
                            save_xp=True)
    conf_dict["brca-n"].t1 = 0.02
    conf_dict["brca-n"].t2 = 0.02

    conf_dict["brca-s"]  = TrainConfig(hidden_dim = 30000, epochs=30, weight_decay=12.0,vertical_decay=0.0, elb_k=1, elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                lambda_c = 200.0, regu_rate=1.08, class_regu_rate=1.08, batch_size = 32,
                            log_interval=100, sparse_regu=0,init_enc="", test=False, lr=0.005, seed=0, model=DiffnapsNet,
                            save_xp=True)
    conf_dict["brca-s"].t1 = 0.02
    conf_dict["brca-s"].t2 = 0.03

    conf_dict["genomes"] = TrainConfig(hidden_dim = 2000, epochs=100, weight_decay=5, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                                lambda_c = 25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=128,
                            log_interval=100, sparse_regu=0, test=False, lr=0.001, model=DiffnapsNet,seed=14401360119984179300,init_enc="bimodal",
                            save_xp=True)
    conf_dict["genomes"].t1=0.03
    conf_dict["genomes"].t2=0.8
    return conf_dict

def get_conf_binaps():
    conf_dict = {}

    conf_dict["cardio"] = TrainConfig(hidden_dim = 500, epochs=50, weight_decay=8.0, elb_k=1, elb_lamb=1, class_elb_k=5, class_elb_lamb=5,
                                lambda_c = 100.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True)
    conf_dict["cardio"].t1 = 0.15
    conf_dict["cardio"].t2 = 0.1


    conf_dict["disease"] = TrainConfig(hidden_dim = 1500, epochs=80, weight_decay=9.0, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                lambda_c = 50, regu_rate=1.08, class_regu_rate=1.08, batch_size=32,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,seed=14401360119984179300,
                            save_xp=True)
    conf_dict["disease"].t1=0.15
    conf_dict["disease"].t2=0.1



    conf_dict["brca-n"] = TrainConfig(hidden_dim = 10000, epochs=20, weight_decay=5.0,vertical_decay=0.0, elb_k=1, elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                lambda_c = 10.0, regu_rate=1.08, class_regu_rate=1.08, batch_size = 32,
                            log_interval=100, sparse_regu=0,init_enc="", test=False, lr=0.005, seed=0, model=DiffnapsNet,
                            save_xp=True)
    conf_dict["brca-n"].t1 = 0.02
    conf_dict["brca-n"].t2 = 0.02

    conf_dict["brca-s"]  = TrainConfig(hidden_dim = 30000, epochs=30, weight_decay=12.0,vertical_decay=0.0, elb_k=1, elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                lambda_c = 200.0, regu_rate=1.08, class_regu_rate=1.08, batch_size = 32,
                            log_interval=100, sparse_regu=0,init_enc="", test=False, lr=0.005, seed=0, model=DiffnapsNet,
                            save_xp=True)
    conf_dict["brca-s"].t1 = 0.02
    conf_dict["brca-s"].t2 = 0.03

    conf_dict["genomes"] = TrainConfig(hidden_dim = 2000, epochs=100, weight_decay=5, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                                lambda_c = 25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=128,
                            log_interval=100, sparse_regu=0, test=False, lr=0.001, model=DiffnapsNet,seed=14401360119984179300,init_enc="bimodal",
                            save_xp=True)
    conf_dict["genomes"].t1=0.03
    conf_dict["genomes"].t2=0.8
    return conf_dict

def get_conf_bc():
    conf_dict = {}

    conf_dict["cardio"] = TrainConfig(hidden_dim = 500, epochs=50, weight_decay=8.0, elb_k=1, elb_lamb=1, class_elb_k=5, class_elb_lamb=5,
                                lambda_c = 100.0, regu_rate=1.08, class_regu_rate=1.08, batch_size=64,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.005, model=DiffnapsNet,seed=14401360119984179300,
                              loss=weightedXorCover, alpha=600,save_xp=True,k_w=10, k_f=7
                            )
    conf_dict["cardio"].t1 = 0.15
    conf_dict["cardio"].t2 = 0.1


    conf_dict["disease"] = TrainConfig(hidden_dim = 1500, epochs=160, weight_decay=9.0, elb_k=0, elb_lamb=0, class_elb_k=0, class_elb_lamb=0,
                                lambda_c = 50, regu_rate=1.08, class_regu_rate=1.08, batch_size=32,
                            log_interval=1000, sparse_regu=0, test=False, lr=0.01, model=DiffnapsNet,seed=14401360119984179300,
                              loss=weightedXorCover, alpha=600,save_xp=True,
                            )
    conf_dict["disease"].t1=0.15
    conf_dict["disease"].t2=0.1



    conf_dict["brca-n"] = TrainConfig(hidden_dim = 10000, epochs=40, weight_decay=5.0,vertical_decay=0.0, elb_k=1, elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                lambda_c = 10.0, regu_rate=1.08, class_regu_rate=1.08, batch_size = 32,
                            log_interval=100, sparse_regu=0,init_enc="", test=False, lr=0.005, seed=0, model=DiffnapsNet,
                              loss=weightedXorCover, alpha=600,save_xp=True,k_w=200, k_f=100
                            )
    conf_dict["brca-n"].t1 = 0.02
    conf_dict["brca-n"].t2 = 0.02

    conf_dict["brca-s"]  = TrainConfig(hidden_dim = 30000, epochs=60, weight_decay=12.0,vertical_decay=0.0, elb_k=1, elb_lamb=1, class_elb_k=1, class_elb_lamb=1,
                                lambda_c = 200.0, regu_rate=1.08, class_regu_rate=1.08, batch_size = 32,
                            log_interval=100, sparse_regu=0,init_enc="", test=False, lr=0.005, seed=0, model=DiffnapsNet,
                              loss=weightedXorCover, alpha=600,save_xp=True,k_w=1000, k_f=500
                            )
    conf_dict["brca-s"].t1 = 0.02
    conf_dict["brca-s"].t2 = 0.03

    conf_dict["genomes"] = TrainConfig(hidden_dim = 2000, epochs=200, weight_decay=5, elb_k=10, elb_lamb=10, class_elb_k=20, class_elb_lamb=20,
                                lambda_c = 25.0, regu_rate=1.1, class_regu_rate=1.1, batch_size=128,
                            log_interval=100, sparse_regu=0, test=False, lr=0.001, model=DiffnapsNet,seed=14401360119984179300,init_enc="bimodal",
                              loss=weightedXorCover, alpha=600,save_xp=True,
                            )
    conf_dict["genomes"].t1=0.03
    conf_dict["genomes"].t2=0.8
    return conf_dict


def expR(output_dir=None, dataset='cardio', seed=0):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("WARNING: Running purely on CPU. Slow.")
    else:
        device = torch.device("cuda")

    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), r"..", "data")
    if dataset == "cardio":
        data, labels, translator = load_cardio_vasc(data_root)
        label_dict = {0:"healthy",1:"heartattack"}
        id=0
    elif dataset == "disease":
        data, labels, label_dict, translator = load_disease_symptom_prediction(data_root)
        translator = list(translator.values())
        id=1
    elif dataset == "brca-n":
        data, labels, translator = load_brca_bin(data_root)
        label_dict = {0:"Adj normal",1:"tumor"}
        id=2
    elif dataset == "brca-s":
        data, labels, translator = load_brca_mult(data_root)
        label_dict = {0:"0",1:"1",2:"2",3:"3"}
        translator = list(translator)
        id=3
    elif dataset == "genomes":
        print("NOT AVAILABLE")
        return
        # data, labels, label_dict, translator = load_1000_genomes(data_root)
        # data, labels, label_dict, translator = load_1000_genomes()
        id=4

    conf_dict_all = {'binaps':get_conf_binaps(), 'diffnaps':get_conf_diff(), 'bc':get_conf_bc()}
    # conf_dict_all = {'binaps':get_conf_binaps()}
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","results","real_results")
    df = pd.DataFrame()
    for method, conf_dict in conf_dict_all.items():

        conf = conf_dict[dataset]
        if seed > 0:
              conf.seed = seed
        log_dir = os.path.join(root, r"runs/")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tag =  f'{current_time}_expR_{method}_db_{dataset}_a{conf.alpha}'
        run_dir = log_dir +  tag 
        writer = SummaryWriter(log_dir=run_dir)
        writer.add_text('Run info', 'Hyperparameters:' + conf.to_str())

        start_time = time.time()
        # Train diffnaps 
        # model, new_weights, trainDS = learn_diffnaps_net(data,conf,labels = labels)
        # Train diffnaps
        if method == "binaps":
                model, new_weights, trainDS, test_loader = learn_xp_binaps(data, conf, labels = labels,ret_test=True, writer=writer, verbose=False)
        else:
                model, new_weights, trainDS, test_loader = learn_diffnaps_net(data, conf, labels = labels,ret_test=True,verbose=False, writer=writer)
                       
        time_taken = time.time() - start_time
        time_taken = time_taken / 60
        enc_w = model.fc0_enc.weight.data.detach().cpu()
        
        if conf.save_xp:
                file = os.path.join(run_dir, "model_weight.pt")
                torch.save(enc_w, file)
                file = os.path.join(run_dir, "data")
                np.save(file,data)
                file = os.path.join(run_dir, "labels")
                np.save(file, labels)

        
        if method == 'binaps':
                _,_,_,num_pat,res_dict, gen_patterns = get_positional_patterns_binaps(weights=enc_w,data=data, labels=labels, general=True)

        else:
                c_w = model.classifier.weight.detach().cpu()
                if conf.save_xp:
                    file = os.path.join(run_dir, "classif_weight.pt")
                    torch.save(c_w, file)
                _,_,_,num_pat,res_dict, gen_patterns = get_positional_patterns(enc_w,c_w, general=True, t_mean=1, t1=conf.t1,t2=conf.t2, device=device)
                       
        # extract the differntial patterns, t1 is t_e and t2 is t_c 
        # _,_,_,num_pat,res_dict, _ = get_positional_patterns(enc_w,c_w, general=True, t_mean=1.0,  t1=conf.t1,t2=conf.t2)
        
        
        metric_result = {'method':method, 'db':dataset, 'ktop':0, 'NMF':'_'}
        metric_result.update(mean_compute_metric(data, labels, res_dict, device=device))
        res_dict_int = {int(k):v for k,v in res_dict.items()}
        line_x, line_y, auc = roc(res_dict_int, data,labels,label_dict,translator,verbose=False)
        metric_result['roc_auc'] = auc
        
        df_metric = pd.DataFrame(metric_result, index=[0])

        for key, value in metric_result.items():
            if key != "method" and key != 'db' and key != 'NMF':
                writer.add_scalar(key, value, id)

        dr = pd.DataFrame({'x':line_x, 'y':line_y})
        dr.to_csv(os.path.join(run_dir, 'auc_roc_data_noNMF.csv'))

        res_to_csv(method, dataset, res_dict_int, data, labels, label_dict, translator, output=run_dir)


        if method == 'bc':
            new_p = compile_new_pat_by_class(labels=labels, patterns=res_dict, data=data, n=[2,3], device=device, max_iter=200, rank=conf.k_f)

            for keyk, val in new_p.items():

                metric_result = {'method':method,'db':dataset, 'ktop':keyk, 'NMF':'filter'}
                metric_result.update(mean_compute_metric(data, labels, val, device=device))
                res_dict_int = {int(k):v for k,v in val.items()}
                line_x, line_y, auc = roc(res_dict_int, data,labels,label_dict,translator,verbose=False)
                metric_result['roc_auc'] = auc
                df_metric = pd.concat([df_metric, pd.DataFrame(metric_result, index=[0])], ignore_index=True)
                
                for key, value in metric_result.items():
                    if key != "method" and key != 'db' and key != 'NMF':
                                writer.add_scalar(key, value, keyk)

                dr = pd.DataFrame({'x':line_x, 'y':line_y})
                dr.to_csv(os.path.join(run_dir, f'auc_roc_data_NMF_filter_{keyk}.csv'))

            # new_p = compile_new_pat_whole(labels=labels, patterns=res_dict, data=data, n=[2,3], device=device, max_iter=2000, rank=conf.k_w)

            # for keyk, val in new_p.items():

            #     metric_result = {'method':method,'db':dataset, 'ktop':keyk, 'NMF':'whole'}
            #     metric_result.update(mean_compute_metric(data, labels, val, device=device))
            #     res_dict_int = {int(k):v for k,v in res_dict.items()}
            #     line_x, line_y, auc = roc(res_dict_int, data,labels,label_dict,translator,verbose=False)
            #     metric_result['roc_auc'] = auc
            #     df = pd.concat([df, pd.DataFrame(metric_result, index=[0])], ignore_index=True)
                
            #     for key, value in metric_result.items():
            #         if key != "method" and key != 'db' and key != 'NMF':
            #                     writer.add_scalar(key, value, keyk)

            #     dr = pd.DataFrame({'x':line_x, 'y':line_y})
            #     dr.to_csv(os.path.join(run_dir, f'auc_roc_data_NMF_whole_{keyk}.csv'))


        write_time(method, dataset,time_taken,res_dict, output_dir=run_dir)
        file = rf"expR_{dataset}_{method}.xlsx"
        df_metric.to_excel(os.path.join(run_dir, file),index=False)
        df = pd.concat([df, df_metric], ignore_index=True)

        if output_dir:
                cleanup(run_dir,os.path.join(output_dir, tag+'.zip'))
            
        del model
        writer.close()

    file = rf"expR_{dataset}.xlsx"
    if output_dir:
            df.to_excel(os.path.join(output_dir, file),index=False)
            print(f"Saved result in {os.path.join(output_dir, file)}")
    else:
            df.to_excel(os.path.join(root, file),index=False)
            print(f"Saved result in {os.path.join(root, file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset",required=True,type=str)
    parser.add_argument("--output_dir",required=False,type=str)
    parser.add_argument("--seed",required=False,type=int, default=0)

    args = parser.parse_args(['-d', 'cardio'])
    # expR(dataset)
    expR(args.output_dir, args.dataset)
