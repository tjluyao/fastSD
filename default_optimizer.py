import torch
from pytorch_lightning import seed_everything

class default_optimazer:
    def __init__(
        self,
        model_name: str,
        batch_option: int = 1,
        max_batch_size: int = 10,
        seed: int = 49,
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        **kwargs
        ):
        seed_everything(seed)
        self.model_name = model_name
        self.device = device
        self.init_model(**kwargs)
        self.max_batch_size = max_batch_size
        self.batch_option = batch_option
        self.wait_preprocess = []
        self.wait_runtime = []
        self.wait_postprocess = []
        pass

    def init_model(self, **kwargs):
        pass

    def select_batch(self, pool):
        batch = []
        pool = sorted(pool, key=lambda req: req.time)

        for item in pool:
            if len(batch) == self.max_batch_size:
                break
            if item.state == 0:
                pass
            batch.append(item)
        return batch
    
    def runtime(self,**kwargs):
        r_batch = []
        batch = self.select_batch(self.wait_runtime)
        if batch:
            for item in batch:
                item.state = 1
                self.wait_runtime.remove(item)
            r_batch = self.iteration(batch, **kwargs)
        if r_batch:
            for item in r_batch:
                if item.state == 2:
                    self.wait_postprocess.append(item)
                else:
                    self.wait_runtime.append(item)

    def check_prepost(self):
        if len(self.wait_preprocess) >= self.batch_option:
            print('Start encoding')
            encode_process = self.wait_preprocess
            self.wait_preprocess = []
            encode_process = self.preprocess(encode_process)

        if len(self.wait_postprocess) >= self.batch_option:
            print('Start decoding')
            decode_process = self.wait_postprocess
            self.wait_postprocess = []
            decode_process = self.postprocess(decode_process)


    def preprocess(self):
        pass

    def iteration(self):
        pass

    def postprocess(self):
        pass

    def save_to_local(self):
        pass