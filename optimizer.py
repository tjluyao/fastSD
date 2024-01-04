import yaml
import time
from punica import KvCache
from pytorch_lightning import seed_everything
from model import llama_model, tokenizer_model, tokenizer_decode_model, visual_model


class Request():
    def __init__(
            self,
            input,
            lora_id = None,
            img_path = None,
        ) -> None:
        self.input = input
        self.time = time.time()
        self.id = self.time
        self.state = 0
        self.token_ids = None
        self.output = None
        self.is_prefill = True
        self.lora_id = lora_id
        self.img_path = img_path

class Optimizer:
    def __init__(
            self, 
            config_file,
            ):
        config = yaml.load(
            stream=open(config_file, 'r'),
            Loader=yaml.FullLoader
            )
        self.config = config
        self.waitlists=[]
        self.batch_size = config.get('batch_size',1)
        self.iteration_batch_size = config.get('iteration_batch_size',1)
        seed = config.get('seed',49)
        seed_everything(seed)

        model_num = config['model_num']
        for i in range(model_num):
            model_name = 'model'+str(i)
            model_config = config[model_name]
            model_type = model_config['model_type']
            if model_type == 'tokenizer':
                model = tokenizer_model(model_config)
            elif model_type == 'tokenizer_decode':
                model = tokenizer_decode_model(model_config)
            elif model_type in ['llama','vituna']:
                model = llama_model(model_config)
            elif model_type == 'visual':
                model = visual_model(model_config)
            else:
                raise ValueError('model_type not supported.')
            setattr(self, model_name, model)
            self.waitlists.append([])
            print(f'{model_name} initialized.')
    
    def select(self,pool,batch_size):
        batch = []
        pool = sorted(pool, key=lambda req: req.time)

        for item in pool:
            if len(batch) == batch_size:
                break
            batch.append(item)
        return batch
    
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
            #print(f'{model_name} running.')
            for item in batch:
                waitlist.remove(item)
            batch = model.run(batch,
                              language_model = getattr(self,'model'+str(model.next_model)) if model.model_type == 'visual' else None,
                              )
            for item in batch:
                if isinstance(item.state,int):
                    self.waitlists[item.state].append(item)

if __name__ == '__main__':
    optimizer = Optimizer(
        'configs/lynx_config.yaml'
        )
    def get_usr_input():
        while True:
            usr_input = input()
            if usr_input != '\n':
                req = Request(
                    usr_input,
                    img_path='inputs/02.jpg',
                    )
                print('input received.\n')
                optimizer.waitlists[0].append(req)

    import threading
    t = threading.Thread(target=get_usr_input)
    t.daemon = True
    t.start()
    while True:
        optimizer.runtime()

