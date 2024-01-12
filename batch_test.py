from datasets import load_dataset
import random
import time
from optimizer import Optimizer, Request
class optimizer_test(Optimizer):
    def __init__(
            self, 
            config_file, 
            dataset_file,
            batch_size = 1,
            iteration_batch_size = 1,
            ):
        super().__init__(config_file)
        dataset = load_dataset(dataset_file)
        self.sentences = dataset['train']['target']
        self.batch_size = batch_size
        self.iteration_batch_size = iteration_batch_size

    def runtime(self):
        for i,waitlist in enumerate(self.waitlists):
            if len(waitlist) == 0:
                continue
            model_name = 'model'+str(i)
            model = getattr(self,model_name)
            batch_size = self.batch_size if model.model_mode == 'batch' else self.iteration_batch_size
            if model.model_mode == 'batch' and len(waitlist) < batch_size:
                continue
            batch = self.select(waitlist,batch_size)
            for item in batch:
                waitlist.remove(item)
            batch = model.run(batch,
                              language_model = getattr(self,'model'+str(model.next_model)) if model.model_type == 'visual' else None,
                              )
            for item in batch:
                if isinstance(item.state,int):
                    self.waitlists[item.state].append(item)
                else:
                    #print(time.time()-item.time)
                    data_log.append(time.time()-item.time)
    def keep_input(self):
        if len(self.waitlists[0]) < self.batch_size:
            choice = random.choice(self.sentences)
            req = Request(choice)
            self.waitlists[0].append(req)

if __name__ == '__main__':
    optimizer = optimizer_test(
        #'configs/lynx_config.yaml'
        'configs/llama_config.yaml',
        dataset_file='Jyshen/Chat_Suzumiya_Haruhi',
        batch_size=20,
        iteration_batch_size=20,
        )
    data_log = []
    while len(data_log) < 200:
        optimizer.keep_input()
        optimizer.runtime()

    with open('data_log.json','w') as f:
        import json
        data = {
                'batch_size':optimizer.batch_size,
                'iteration_batch_size':optimizer.iteration_batch_size,
                'data':data_log,
        }
        json.dump(data,f,indent=4)
