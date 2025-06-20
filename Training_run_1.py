import subprocess as sp

import trener2
import utilities

from pathlib import Path

Current_Directory = Path(__file__).parent
print(Current_Directory)

Model_Save_Dir = Current_Directory /'Model_saves'
Data_Save_Dir = Current_Directory /'Saved_datasets'
Run_save_dir = Current_Directory / 'Run_data'


def main():
    utilities.text_export(f"\nTraining run: {utilities.Get_time()} ","Training_runs",Run_save_dir)
    
    utilities.text_export(f"\nRun 1\nTrener: --hidden=16,32,64,32,16 --steps_per_batch=260 --batches=350 --samples_per_batch=300 --tau=False --sample_buffer=0.5 --theta=True --multiprocessing=True --use_cuda=True ","Training_runs",Run_save_dir)
    sp.run(["python","trener2.py","--hidden=16,32,64,32,16","--steps_per_batch=260","--batches=350","--samples_per_batch=300","--sample_buffer=0.5","--theta=True", "--multiprocessing=True"])

    
    '''
    utilities.text_export(f"\nRun 2\nTrener: --hidden=16,32,64,32,16 --steps_per_batch=260 --batches=350 --samples_per_batch=300 --tau=False --sample_buffer=0.5 --theta=True --multiprocessing=True --noise=0.25","Training_runs",Run_save_dir)
    sp.run(["python","trener2.py","--hidden=16,32,64,32,16","--steps_per_batch=260","--batches=350","--samples_per_batch=300","--sample_buffer=0.5","--theta=True","--multiprocessing=True","--noise=0.25"])

    utilities.text_export(f"\nRun 3\nTrener: --hidden=16,32,64,32,16 --steps_per_batch=260 --batches=350 --samples_per_batch=300 --tau=False --sample_buffer=0.5 --theta=True --multiprocessing=True --noise=0.3","Training_runs",Run_save_dir)
    sp.run(["python","trener2.py","--hidden=16,32,64,32,16","--steps_per_batch=260","--batches=350","--samples_per_batch=300","--sample_buffer=0.5","--theta=True","--multiprocessing=True","--noise=0.3"])

    utilities.text_export(f"\nRun 4\nTrener: --hidden=16,32,64,32,16 --steps_per_batch=260 --batches=350 --samples_per_batch=300 --tau=False --sample_buffer=0.5 --theta=True --multiprocessing=True --noise=0.35","Training_runs",Run_save_dir)
    sp.run(["python","trener2.py","--hidden=16,32,64,32,16","--steps_per_batch=260","--batches=350","--samples_per_batch=300","--sample_buffer=0.5","--theta=True","--multiprocessing=True","--noise=0.35"])
    
    utilities.text_export(f"\nRun 5\nTrener: --hidden=16,32,64,32,16 --steps_per_batch=260 --batches=350 --samples_per_batch=300 --tau=False --sample_buffer=0.5 --theta=True --multiprocessing=True --noise=0.4","Training_runs",Run_save_dir)
    sp.run(["python","trener2.py","--hidden=16,32,64,32,16","--steps_per_batch=260","--batches=350","--samples_per_batch=300","--sample_buffer=0.5","--theta=True","--multiprocessing=True","--noise=0.4"])
    '''
    

#TEST WITH A BIGGER DATASETT, CHECK NUMBER OF LOGIC CORES # Try Max workers os.process_cpu_count()-4
    
    
    
    


if __name__=="__main__":
    main()