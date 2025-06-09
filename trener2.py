import torch as th
import argparse
import numpy as np
import torch.cuda
from time import time
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
import Data_gen2
import Data_gen2_theta
import models 
import utilities

"""
Ad function to help argument parsing, a workaround to make an in terminal definition of hidden work.
"""
def int_list(arg):
    return list(map(int, arg.split(',')))

"""
Parse all arguments for later use 
get_arguments
"""
def get_arguments():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--input_size',default=2,type=int,help='Number of values inputed into the neural network')
    parser.add_argument('--hidden',default=[32,64,32],type=int_list,help='Number of hidden layers as well as their size')
    parser.add_argument('--opti',default = 'AdamW', type = str,help='choose optimizer (use full string from: https://pytorch.org/docs/main/optim.html)')

    parser.add_argument('--tau',default=False,type=bool,help= 'use torque in the dynamics')
    parser.add_argument('--theta',default=False,type=bool,help= 'use q = L*theta instead of q = [x,y].T')

    parser.add_argument('--t_span', default = [0,3], type = list, help = 'span of time used in trajectory generation')
    parser.add_argument('--t_scale', default = 10, type = int, help = 'number of datapoints per trajectory')
    parser.add_argument('--noise', default = 0.1, type = float, help = 'noise to add to input to avoid overfitting')
    parser.add_argument('--tolerance', default = 1e-10, type = int, help = 'tolerance in the solve ivp step of trajectory generation')


    parser.add_argument('--model_save_dir', default=Model_Save_Dir, type=str, help='where to save the trained model')
    parser.add_argument('--run_save_dir', default=Run_save_dir, type=str, help='where to save the data')
    
    parser.add_argument('--save_dataset', default=False, type=bool, help='Do you want to save the dataset?')
    parser.add_argument('--load_dataset', default=None, type=str, help = 'What dataset file to load,')
    parser.add_argument('--data_save_dir', default=Data_Save_Dir, type=str, help='where to save a dataset')


    #parser.add_argument('--verbose', dest='verbose', action='store_true', help='verbose?')
    parser.add_argument('--verbose', default = True, type = bool, help='verbose?')
    parser.add_argument('--seed', default=None, type=int, help='Select seed')
    
    
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-3, type = float, help = 'how fast the weigths decay as to lessen the chance of overfitting and overshoots')
    parser.add_argument('--nonlinearity', default='tanh', type=str, help='neural net nonlinearity')
    
    
    parser.add_argument('--steps_per_batch', default=250, type=int, help='number of gradient steps between prints')
    parser.add_argument('--batches', default=200, type= int, help='How many batches of data to use')
    #parser.add_argument('--total_steps', default=200, type=int, help='number of gradient steps')
    parser.add_argument('--test_ratio', default = 0.5, type = float, help = 'How big should the testing set be in comparison to the training set')

    parser.add_argument('--samples_per_batch',default = 250, type = int, help = 'How many trajectories generated as part of the dataset' )
    parser.add_argument('--samples', default = 20000, type= int, help='placeholder for making reprograming to samples per batch easier')
    parser.add_argument('--sample_buffer',default=0.5,type=float,help='buffer percentage to lessen the chance of overfitting due to the dataset containging all the values')
   
    #parser.add_argument('--baseline', dest='baseline', action='store_true', help='run baseline or experiment?')
    parser.add_argument('--baseline',default=False,type=bool, help="run a basic nn for comparison?")
    parser.add_argument('--name', default='Graydanus-HNN', type=str, help='only this option right now')
    
    parser.add_argument('--use_cuda',default= True, type=bool,help='Run with cuda compatible graphics processor?')
    parser.add_argument('--multiprocessing', default=False,type=bool, help='Use multiprocessing to let the computer use more than one logic core to do the data generation work?')

    parser.add_argument('--Loss_type', default = 'MSE', type = str, help = 'Choose L2 or MSE loss type')

    

    return parser.parse_args()

"""
training function
"""
def trener(args):
    print(args)
    #set randomizer seed
    if args.seed==None:
        args.seed = np.random.randint(0,np.iinfo(np.int32).max)
            
    th.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    
    #Theta or xy Needs rewriting to handle more complex systems.
    if args.theta == False:
        #XY
        args.input_size = 4
    else:
        
        #Theta
        args.input_size = 2


    #Calculating total number of samples needing tto be made
    print("Calculating samples")

    

    test_batch_size = int(args.samples_per_batch * args.test_ratio)
   


    if test_batch_size<1:
        test_batch_size = 1
        print(f"Since {test_batch_size}<1, it has been set to a value of 1")

    train_samples = int(args.batches * args.samples_per_batch*(1+args.sample_buffer))
    test_samples = int(args.batches * test_batch_size*(1+args.sample_buffer))
    args.samples = train_samples + test_samples

    

    
    print(f"Total number of samples: {args.samples}")
    print(f"Samples in train batches:{train_samples}")
    print(f"Samples in test batches: {test_samples}")



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

    #optimizer

    optimizer = utilities.choose_optimizer(model,args)


    if args.Loss_type == 'L2':
        loss_calc = utilities.L2
    else:
        loss_calc = utilities.MSE


    #get/load dataset, save if wanted

    print("Dataset")
    
        



    if args.load_dataset!=None:
        print("Loading dataset")
        data = utilities.loaddatadict(args.load_dataset,args.data_save_dir)
        assert len(data['qp'][0])==args.input_size,"The loaded dataset does not have the correct number of input values"
        assert len(data['qp'])==args.samples,"The loaded dataset is to small for the current batch, batchsize, testsize, and buffer parameters"
        print("Dataset succesfully loaded")
    else:
        if args.theta == False:
            data = Data_gen2.dataset(multiprocessing=args.multiprocessing, samples=args.samples, split_point = train_samples,seed=args.seed,t_span=args.t_span,t_scale=args.t_scale,noise=args.noise,tolerance=args.tolerance, tau=args.tau )
        else:
            data = Data_gen2_theta.dataset_theta(multiprocessing=args.multiprocessing, samples=args.samples, split_point = train_samples,seed=args.seed,t_span=args.t_span,t_scale=args.t_scale,noise=args.noise,tolerance=args.tolerance, tau=args.tau )
    
    if args.save_dataset:
        distinguisher = args.name + str(np.iinfo(np.int32).max)
        print("Saving Dataset")
        utilities.savedatadict(data,f"Dataset {args.name} {distinguisher}", args.data_save_dir)
    
    #arrange dataset
    print("Arranging dataset and loading to Tensor")

    
    batch_list = []
    final_batch = []
    #Divide into train data into batches
    for b in range(args.batches):
        x_list = []
        dx_list = []
        test_x_list = []
        test_dx_list = []

        
        for s in range(args.samples_per_batch):
            length = len(data['qp'])
            i = np.random.randint(0,length)

            x = data['qp'][i]
            dx = data['dqp'][i]

            x_list.append(x)
            dx_list.append(dx)

            np.delete(data['qp'],i)
            np.delete(data['dqp'],i)
        

        for t in range(test_batch_size):
            length = len(data['test_qp'])
            j = np.random.randint(0,length)

            test_x = data['test_qp'][j]
            test_dx = data['test_dqp'][j]

            test_x_list.append(test_x)
            test_dx_list.append(test_dx)

            np.delete(data['test_qp'],j)
            np.delete(data['test_dqp'],j)
        


        x_list = np.asarray(x_list)
        dx_list = np.asarray(dx_list)
        test_x_list = np.asarray(test_x_list)
        test_dx_list = np.asarray(test_dx_list)

        
        if args.use_cuda:
            xy = th.tensor(x_list,requires_grad=True,dtype=torch.float32).cuda()
            test_xy = th.tensor(test_x_list,requires_grad=True, dtype=torch.float32).cuda()
            dxy = th.Tensor(dx_list).cuda()
            test_dxy = th.Tensor(test_dx_list).cuda()
        else:
            xy = th.tensor(x_list,requires_grad=True,dtype=torch.float32)
            test_xy = th.tensor(test_x_list,requires_grad=True, dtype=torch.float32)
            dxy = th.Tensor(dx_list)
            test_dxy = th.Tensor(test_dx_list)

        batch = {'xy':xy,'dxy':dxy,'test_xy':test_xy,'test_dxy':test_dxy}
        if b ==0:
            final_batch.append(batch)
        else:
            batch_list.append(batch)

    






    # setting up stats dict
    print("Stat prep")

    stats = {'arguments': vars(args),'train_loss':[],'test_loss':[]}

    #Basic training loop
    print("Begin Training")
    t0 = time()
    step = 0
    batch_count = 0
    for batch in batch_list:
        batch_count +=1
        for st in range(args.steps_per_batch):
            step+=1
            #Train step
            dxy_hat = model.forward(batch['xy'])
            loss = loss_calc(batch['dxy'],dxy_hat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # test step
            test_dxy_hat = model.forward(batch['test_xy'])
            test_loss = loss_calc(batch['test_dxy'],test_dxy_hat)

            #Log result
            stats['train_loss'].append(loss.item())
            stats['test_loss'].append(test_loss.item())

        #Print Batch result
        print(f"Batches finished: {batch_count}, Step: {step}\n Train loss: {loss.item():.4e}, Test loss: {test_loss.item():.4e}")
        t1 = time()
        frac = (batch_count/len(batch_list))
        percentage = frac*100
        time_elapsed = t1-t0 
        time_from_start_to_end = time_elapsed/frac
        time_til_end = time_from_start_to_end-time_elapsed
        print(f"Time elapsed: {time_elapsed/60:.2f} min. Estimated time until finished: {time_til_end/60:.0f} min")

    #Final loss calculation and variance calulation

    fin_xy = final_batch[0]['xy']
    fin_dxy = final_batch[0]['dxy']
    fin_test_xy = final_batch[0]['test_xy']
    fin_test_dxy = final_batch[0]['test_dxy']

    dxy_hat = model.forward(fin_xy)
    test_dxy_hat = model.forward(fin_test_xy)

    if args.Loss_type == 'L2':
        dist = (fin_dxy-dxy_hat).pow(2)
        loss = dist.sum().sqrt()
        test_dist = (fin_test_dxy-test_dxy_hat).pow(2)
        test_loss = test_dist.sum().sqrt()
    else:
        dist = (fin_dxy-dxy_hat).pow(2)
        loss = dist.mean()
        test_dist = (fin_test_dxy-test_dxy_hat).pow(2)
        test_loss = test_dist.sum().mean()


    loss_std = dist.std()#.item()
    test_loss_std = test_dist.std()#.item()

    stats['train_loss'].append(loss.item())
    stats['test_loss'].append(test_loss.item())
    print("")
    print(f"Final Train loss: {loss:.4e} +/- {loss_std:.4e}")
    print(f"Final Test loss: {test_loss:.4e} +/- {test_loss_std:.4e}")

    #send to txt for overwiew

    info = f"\n---{utilities.Get_time()}---\nFinal Train loss: {loss} +/- {loss_std}\nFinal Test loss: {test_loss} +/- {test_loss_std}"
    utilities.text_export(info,"Training_runs",Run_save_dir)

    #print(stats['arguments'])
    utilities.savedatadict(stats,f"Args_for_model_{args.name}_time_{utilities.Get_time()}",args.run_save_dir)

    return model









"""
Main function to run when train.py is called
"""
if __name__ == "__main__":
    t_start = time()
    arguments = get_arguments()
    model = trener(arguments)

    #save model
    utilities.Save_model(model,f"Model_{arguments.name}_time_{utilities.Get_time()}",arguments.model_save_dir)
    t_end = time()
    t_tot = t_end-t_start
    t_m = t_tot//60
    t_s =  t_tot - t_m*60 
    utilities.text_export(f"\nTotal time elapsed for program: {t_m:.0f} m {t_s:.2f} s","Training_runs",Run_save_dir)
    print(f"\nTotal time elapsed for program: {t_m:.0f} m {t_s:.2f} s")
    
