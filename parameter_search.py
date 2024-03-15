import os
import pathlib
from multiprocessing import Process, Manager
import itertools
import random       
import gc
import time
from tqdm import tqdm

# Define your parameter values as a dictionary
parameters = [ {
    'n_filters': [64],
    'pooling_steps': [4],
    'layers': [1],
    'batch_size': [4],
    'swap_val': [False],
    'combine_val': [True],
    'position': [0.5],
    'quantiles':[False], 
    'transfer':[True,False],
    'ch_org': [True],
    'ch_diff': [True],
    'ch_bg': [True],
    'project':['intercracktransfer'] 
},] 

parameters += parameters
parameters += parameters

def _train(parameter_combination):
    from train import train 
    #train(**parameter_combination)
    while parameter_combination['batch_size'] > 0:
        print(f"Run:", parameter_combination)
        try:
            p = Process(target=train, kwargs=parameter_combination)
            p.start()
            p.join()     
            if p.exitcode == 0:
                p.close()
                gc.collect() 
                break     
            p.close()
            gc.collect() 
            parameter_combination['batch_size'] = parameter_combination['batch_size'] // 2
        except Exception as e:
            print(e)
            break
        
def train_combinations(device, combinations, counter):
    for combination in combinations:
        parameter_combination = dict(zip(parameters[0].keys(), combination))
        
        # add device to parameter_combination
        parameter_combination['DEVICE'] = device
        
        # Perform your testing with the current combination of parameters
        _train(parameter_combination)
        
        # Increment the shared counter
        counter.value += 1
        
def main():
    with Manager() as manager:
        counter = manager.Value('i', 0)
        
        # Generate all combinations
        all_combinations = [] 
        for params in parameters:
            all_combinations.extend(list(itertools.product(*params.values())))

        # Shuffle the list of combinations randomly
        random.shuffle(all_combinations)
        
        half_len = len(all_combinations) // 2
        combinations1 = all_combinations[:half_len]
        combinations2 = all_combinations[half_len:]
        print(half_len)
        
        #train_combinations(0,combinations1, counter)
        # Create processes for parallel training with tqdm progress bar
        process1 = Process(target=train_combinations, args=(0, combinations1, counter))
        process2 = Process(target=train_combinations, args=(1, combinations2, counter))

        # Start the processes
        process1.start()
        process2.start()
        
        # Create a tqdm progress bar and update it with the shared counter
        with tqdm(total=len(all_combinations)) as pbar:
            while counter.value < pbar.total:
                pbar.update(counter.value - pbar.n)
                time.sleep(1)  # pause for 100 milliseconds

        # Wait for both processes to finish
        process1.join()
        process2.join()

if __name__ == '__main__':
    main()