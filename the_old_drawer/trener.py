import torch as th
import argparse
import numpy as np
import torch.cuda

"""
Check and print current dirrectory in case file is going to end up in wrong dirrectory.
Make sure it can save load and store files correctly
"""
from pathlib import Path

Current_Directory = Path(__file__).parent
print(Current_Directory)

Model_Save_Dir = Current_Directory /'Model_saves'
Data_Save_Dir = Current_Directory /'Saved_datasets'
Run_save_dir = Current_Directory / 'Run_data'


"""
Import data generation, models and utilities
"""
import the_old_drawer.Data_gen as Data_gen
import models 
import utilities



"""
Parse all arguments for later use 
get_arguments
"""
def get_arguments():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_size',default=4,type=int,help='Number of values inputed into the neural network')
    parser.add_argument('--hidden',default=[32,64,32],type=list,help='Number of hidden layers as well as their size')
    
    parser.add_argument('--samples',default = 20000, type = int, help = 'How many trajectories generated as part of the dataset' )
    parser.add_argument('--split_ratio', default = 0.5, type = float, help = 'Percentage (in decimals) of the dataset is chosen to be the training data.')

    parser.add_argument('--t_span', default = [0,3], type = list, help = 'span of time used in trajectory generation')
    parser.add_argument('--t_scale', default = 10, type = int, help = 'number of datapoints per trajectory')
    parser.add_argument('--noise', default = 0.1, type = int, help = 'noise to add to input to avoid overfitting')
    parser.add_argument('--tolerance', default = 1e-10, type = int, help = 'tolerance in the solve ivp step of trajectory generation')


    parser.add_argument('--model_save_dir', default=Model_Save_Dir, type=str, help='where to save the trained model')
    parser.add_argument('--run_save_dir', default=Run_save_dir, type=str, help='where to save the data')
    
    parser.add_argument('--save_dataset', default=False, type=bool, help='Do you want to save the dataset?')
    parser.add_argument('--load_dataset', default=None, type=str, help = 'What dataset file to load,')
    parser.add_argument('--data_save_dir', default=Data_Save_Dir, type=str, help='where to save a dataset')


    #parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--verbose', default = True, type = bool, help='verbose?')
    parser.add_argument('--seed', default=None, type=int, help='Select seed')
    
    
    parser.add_argument('--learn_rate', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type = float, help = 'how fast the weigths decay as to lessen the chance of overfitting and overshoots')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    
    parser.add_argument('--total_steps', default=20000, type=int, help='number of gradient steps')
    parser.add_argument('--print_every', default=200, type=int, help='number of gradient steps between prints')
    
    #parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--baseline',default=False,type=bool, help="run a basic nn for comparison?")
    parser.add_argument('--name', default='Graydanus-HNN', type=str, help='only this option right now')
    
    parser.add_argument('--use_cuda',default= True, type=bool,help='Run with cuda compatible graphics processor?')

    parser.add_argument('--Loss_type', default = 'MSE', type = str, help = 'Choose L2 or MSE loss type')


    return parser.parse_args()

"""
training function
"""
def trener(args):
    #set randomizer seed
    if args.seed==None:
        args.seed = np.random.randint(0,np.iinfo(np.int32).max)
            
    th.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    

    #initialize model, optimizer and loss
    print("Initialization")
    if args.verbose:
        print(f"Training {args.name} model:")

    output_size = args.input_size if args.baseline else 2

    if args.baseline:
        if args.use_cuda:
            model = models.BasicNN(args.input_size,output_size,args.nonlinearity,args.hidden).cuda() 
        else:
            model = models.BasicNN(args.input_size,output_size,args.nonlinearity,args.hidden)
    else: 
        if args.use_cuda:
            model = models.HNNG(args.input_size, output_size, args.nonlinearity, args.hidden).cuda()
        else:
            model = models.HNNG(args.input_size, output_size, args.nonlinearity, args.hidden,use_cuda=False )

    optimizer = th.optim.AdamW(model.parameters(),args.learn_rate,weight_decay=args.weight_decay)

    if args.Loss_type == 'L2':
        loss_calc = utilities.L2
    else:
        loss_calc = utilities.MSE

    #get/load dataset, save if wanted
    print("Dataset")
    if args.load_dataset!=None:
        print("Loading dataset")
        data = utilities.loaddatadict(args.load_dataset,args.data_save_dir)
        assert len(data)==4,"The loaded dataset does not contain the correct number of objects"
        print("Dataset succesfully loaded")
    else:
        data = Data_gen.dataset(seed=args.seed,t_span=args.t_span,t_scale=args.t_scale,noise=args.noise,tolerance=args.tolerance )

    if args.save_dataset:
        print("Saving Dataset")
        distinguisher = input("Add something to avoid identical datafile name: ")
        utilities.savedatadict(data,f"Dataset {args.name} {distinguisher}", args.data_save_dir)
    
    #arrange dataset
    print("Arranging dataset and loading to Tensor")
    if args.use_cuda:
        xy = th.tensor(data['qp'],requires_grad=True,dtype=torch.float32).cuda()
        test_xy = th.tensor(data['test_qp'],requires_grad=True, dtype=torch.float32).cuda()
        dxy = th.Tensor(data['dqp']).cuda()
        test_dxy = th.Tensor(data['test_dqp']).cuda()
    else:
        xy = th.tensor(data['qp'],requires_grad=True,dtype=torch.float32)
        test_xy = th.tensor(data['test_qp'],requires_grad=True, dtype=torch.float32)
        dxy = th.Tensor(data['dqp'])
        test_dxy = th.Tensor(data['test_dqp'])

    # setting up stats dict
    print("Stat prep")

    

    stats = {'arguments': vars(args),'train_loss':[],'test_loss':[]}

    #Basic training loop
    print("Begin Training")

    for step in range(args.total_steps+1):

        #Train step
        dxy_hat = model.forward(xy)
        loss = loss_calc(dxy,dxy_hat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # test step
        test_dxy_hat = model.forward(test_xy)
        test_loss = loss_calc(test_dxy,test_dxy_hat)

        #Log result
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())

        #Print result
        if args.verbose and step % args.print_every == 0:
            print(f"step: {step}, Train loss: {loss.item():.4e}, Test loss: {test_loss.item():.4e}")


    #Final loss calculation and variance calulation

    dxy_hat = model.forward(xy)
    test_dxy_hat = model.forward(test_xy)

    if args.Loss_type == 'L2':
        dist = (dxy-dxy_hat).pow(2)
        loss = dist.sum().sqrt()
        test_dist = (test_dxy-test_dxy_hat).pow(2)
        test_loss = test_dist.sum().sqrt()
    else:
        dist = (dxy-dxy_hat).pow(2)
        loss = dist.sum().sqrt()
        test_dist = (test_dxy-test_dxy_hat).pow(2)
        test_loss = test_dist.sum().sqrt()


    loss_std = dist.std()#.item()
    test_loss_std = test_dist.std()#.item()


    print(f"Final Train loss: {loss.item():.4e} +/- {loss_std:.4e}")
    print(f"Final Test loss: {test_loss.item():.4e} +/- {test_loss_std:.4e}")

    #print(stats['arguments'])
    utilities.savedatadict(stats,f"Args_for_model_{args.name}_time_{utilities.Get_time()}",args.run_save_dir)

    return model









"""
Main function to run when train.py is called
"""
if __name__ == "__main__":
    arguments = get_arguments()
    model = trener(arguments)

    #save model
    utilities.Save_model(model,f"Model_{arguments.name}_time_{utilities.Get_time()}",arguments.model_save_dir)
