

from dataclasses import dataclass

@dataclass
class Model:
    name: str =  'LODA (5, 10)'
    type: str =  'LODA'
    pars: tuple =  (5, 10)


@dataclass
class Model:
    name: str =  "LOF (70, 'euclidian')"
    type: str =  'LOF'
    pars: tuple =  (70, 'euclidean')



if __name__ == '__main__':
    
    
    
    model_list = [Model('LODA (5, 10)', 'LODA', (5,10)), Model('Iforest (10, 0.2)', 'Iforest', (10,0.2))]
    
    data_list = [('Annthyroid', 'annthyroid.mat'), ('Arrhythmia', 'arrhythmia.mat')]
    
    train_from_scratch(model_list, data_list)