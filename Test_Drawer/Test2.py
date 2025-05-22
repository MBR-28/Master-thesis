import the_old_drawer.Data_gen as gen
import utilities as ut
from pathlib import Path


Current_Directory = Path(__file__).parent
print(Current_Directory)

Model_Save_Dir = Current_Directory /'Model_saves'
Data_Save_Dir = Current_Directory /'Saved_datasets'
Run_save_dir = Current_Directory / 'Run_data'


dat = gen.dataset(samples=3)
ut.savedatadict(dat,"test2", Data_Save_Dir)

#da = ut.loaddatadict("test2",Data_Save_Dir)
#print(type(da['qp']))