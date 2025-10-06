from si.base.transformer import Transformer
from si.data.dataset import Dataset
import numpy as np

class VarianceThreshold (Transformer):
    
    def __init__(self,threshold: float = 0.4, **kwargs ):
        
        self.threshold =  threshold
        self.variance = None
        
    def _fit(self, dataset: Dataset) -> "VarianceThreshold":
        
        self.variance = np.var(dataset.X, axis = 0) # axis = 0 vai estimar a variancia ao longo das linhas, se fosse 1 ia computar a variancia para cada linha
        
        
    def _transform(self, dataset: Dataset) -> Dataset:
        
        mask = self.variance >= self.threshold
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask]
        
        
        return Dataset(X=X, features=features, y=dataset.y, label=dataset.label)
        
        
