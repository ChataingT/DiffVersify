from method.main import main


if __name__ == '__main__':

    arg = r"-i C:\Users\chataint\Documents\projet\diffversify\runs\data.dat -o C:\Users\chataint\Documents\projet\diffversify\runs\output --batch_size 20000 --test_batch_size 20000 --epochs 50 --log_interval 1000 --lr 0.001"

    main(arg.split(' '))