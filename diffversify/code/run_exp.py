import argparse

from exp2 import exp2
from exp1 import exp1
from exp3 import exp3
from exp4 import exp4
from real_exp import expR

dxp = {
    1:exp1,
    2:exp2,
    3:exp3,
    4:exp4,
    5:expR
}
if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='Process some input argument.')
      parser.add_argument('--output', type=str, help='copy output to this dir', default=None)
      parser.add_argument('--xp', type=int, help='run xp number', default=1)
      parser.add_argument("-d","--dataset",required=False,type=str)
      parser.add_argument("--seed",required=False,type=int, default=0)
    
      args = parser.parse_args()

      dxp[args.xp](output_dir=args.output, dataset=args.dataset, seed=args.seed)
    
