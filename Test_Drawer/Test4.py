import numpy as np
import utilities as ut

from pathlib import Path

Current_Directory = Path(__file__).parent
print(Current_Directory)

Model_Save_Dir = Current_Directory /'Model_saves'
Data_Save_Dir = Current_Directory /'Saved_datasets'
Run_save_dir = Current_Directory / 'Run_data'

stats = {'a':[1,2,3,4]}
name = "BOB"

ut.savedatadict(stats,f"Stats_{name}_time_{ut.Get_time()}",Run_save_dir)


#print(np.iinfo(np.int32).max)