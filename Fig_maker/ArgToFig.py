import numpy as np
import matplotlib.pyplot as plt
import sys

from pathlib import Path
Current_Directory = Path(__file__).parent
Par_Dir = Current_Directory.parent
Fig_Save_Dir = Par_Dir /'Fig'
Run_save_dir = Par_Dir / 'Run_data'

PD = str(Par_Dir.absolute())
print(PD)
sys.path.append(PD)

from  utilities  import loadargsdict as ld



def TimeNameToFig(time,name,num):

    arg_name = f"Args_for_model_{name}_time_{time}"
    data = ld(arg_name,Run_save_dir)

    args = data["arguments"]
    train = data["train_loss"]
    test = data["test_loss"]


    #Deleting personal
    del args["data_save_dir"]
    del args["model_save_dir"]
    del args["run_save_dir"]
    if args["load_dataset"]==None:
        del args["load_dataset"]
    
    #Table

    '''
    coll = ("Hyperparameter","Value")
    cell_data = []
    for i in args:
        cell_data.append([i,args[i]])

    plt.figure(figsize=(10,10))

    table = plt.table(cellText=cell_data,
                      colLabels=coll,
                      loc='bottom',
                      bbox=[0,-3.5,1,3])
    '''
    #Print to latex table
    with open(Fig_Save_Dir/f"{name}.txt","a") as f:
        f.write(time)
        f.write("\n")
        f.write("\\begin{table}[h!]")
        f.write("\n")
        f.write("\\centering")
        f.write("\n")
        f.write("\\begin{tabular}{|c|c|}")
        f.write("\n")
        f.write("\t\\hline")
        f.write("\n")
        for i in args:
            f.write(f"\t{i} & {args[i]}\\\\")
            f.write("\n")
            f.write("\t\\hline")
            f.write("\n")
        
        f.write("\\end{tabular}")
        f.write("\n")
        f.write("\\caption{Hyperparameters of model ")
        f.write(f"{num}")
        f.write("}")
        f.write("\n")
        f.write("\\label{table:mod")
        f.write(f"{num}")
        f.write("}")
        f.write("\n")
        f.write("\\end{table}")
        f.write("\n")
        f.write("\n")

                      

    #Loss graph
    
    comp_train = np.array([])
    comp_test = np.array([])

    batch_size = args["samples_per_batch"] 
    comp = 1 #Plot every x batchess to make the graph smaller

    for j in range(len(train)):
        if j%(batch_size*comp) == 0 and j!=0:
            comp_train = np.append(comp_train,np.log10(np.mean(train[j-batch_size:j])))
            comp_test = np.append(comp_test,np.log10(np.mean(test[j-batch_size:j])))
        elif j%(batch_size*comp) == 0:
            comp_train = np.append(comp_train,np.log10(train[j]))
            comp_test = np.append(comp_test,np.log10(test[j]))

    comp_train = np.append(comp_train,np.log10(train[-1]))
    comp_test = np.append(comp_test,np.log10(test[-1]))


    x = np.linspace(0,len(comp_train)-1,len(comp_train))*comp

    
    plt.plot(x,comp_train,'r-',label='Training loss')
    plt.plot(x,comp_test,'b-',label='Test loss')


    plt.xlabel('Training_batches')
    plt.ylabel('MSE Loss (Log10)')
    plt.ylim(-4.5,2)

    plt.title(f"Loss curve model {num}")
    
    plt.legend()

    plt.savefig(fname=Fig_Save_Dir/f"{name}_{time}.png",dpi=300)
    plt.close()
    #plt.show()











if __name__ == "__main__":
    TimeNameToFig("2025_05_07_13_36_59","Graydanus-HNN",1)
    TimeNameToFig("2025_05_07_14_52_26","Graydanus-HNN",2)
    TimeNameToFig("2025_05_07_17_12_36","Graydanus-HNN",3)
    TimeNameToFig("2025_05_07_19_04_45","Graydanus-HNN",4)
    TimeNameToFig("2025_05_26_14_51_54","Graydanus-HNN",5)
    TimeNameToFig("2025_05_30_20_35_46","Graydanus-HNN",6)
    TimeNameToFig("2025_06_05_11_32_06","Graydanus-HNN",7)
    TimeNameToFig("2025_06_05_17_24_26","Graydanus-HNN",8)
    TimeNameToFig("2025_06_05_21_39_59","Graydanus-HNN",9)
    TimeNameToFig("2025_06_05_22_43_08","Graydanus-HNN",10)
