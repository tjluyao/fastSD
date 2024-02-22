'''
Some Bugs in Mutiprocessing using python & cuda
Maybe need C/C++
'''

from optimizer import Optimizer, yaml, seed_everything
from sd_helper import get_condition, load_model, unload_model, save_video_as_grid_and_mp4, torch, EDMSampler, autocast, perform_save_locally, append_dims, VideoDecoder
import random, time, copy, os
import multiprocessing as mp
from sd_optimizer import sd_request
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
mp.set_start_method('spawn',force=True)

class sd_optimizer_muti(Optimizer):
    def __init__(self, config_file):
        config = yaml.load(
            stream=open(config_file, 'r'),
            Loader=yaml.FullLoader
            )
        
        self.config = config
        self.waitlists=[[],[],[]]
        self.batch_configs = config['batch_configs']
        seed = config.get('seed',49)
        seed_everything(seed)

        self.model_name = config['model_name']
        model_config = config['model']
        from sd_helper import init_model, init_sampling
        self.lowvram_mode = config.get('lowvram_mode',False)
        self.dtype_half = True if config.get('dtype', None) == 'half' else False

        print('Loading model'+' in half' if self.dtype_half else '')
        state = init_model(model_config, load_filter=True, dtype_half=self.dtype_half)
        steps = config.get('default_steps',10)
        state['W'] = model_config.get('W', 1024)
        state['H'] = model_config.get('H', 1024)
        state['C'] = model_config.get('C', 4)
        state['F'] = model_config.get('F', 8)
        state['T'] = 6 if self.model_name in ['svd'] else None
        state['options'] = model_config.get('options', {})
        state['options']["num_frames"] = state['T']
        state['sampler'] = init_sampling(options=state['options'],steps=steps)
        state['img_sampler'] = init_sampling(options=state['options'],img2img_strength=0.75,steps=steps)
        state['saving_fps'] = 6
        self.state = state

        device_count = torch.cuda.device_count()
        device_count = 4
        self.num_decoding = device_count // 4
        self.num_sampling = device_count - self.num_decoding

        self.sampling_status = [None] * self.num_sampling
        self.decoding_status = [None] * self.num_decoding

        for i in range(device_count):
            device = torch.device('cuda:'+str(i))
            if i == 0:
                self.sampling_model_0 = (self.state['model'].model.cuda(),self.state['model'].denoiser.cuda())
            elif i < self.num_sampling:
                model = copy.deepcopy(self.state['model'].model).to(device)
                denoiser = copy.deepcopy(self.state['model'].denoiser).to(device)
                setattr(self,'sampling_model_'+str(i),(model,denoiser))
            elif i == self.num_sampling:
                self.decoding_model_0 = self.state['model'].first_stage_model.to(device)
            else:
                decoding_model = copy.deepcopy(self.state['model'].first_stage_model).to(device)
                setattr(self,'decoding_model_'+str(i-self.num_sampling),decoding_model)
        
        self.state['model'].conditioner.cuda()
        print('Model loaded')

    def runtime(self):
        for i,waitlist in enumerate(self.waitlists):
            if len(waitlist) == 0:
                continue

            batch_size = self.batch_configs[i]
            if i == 0:
                batch = self.select(waitlist,batch_size)
                for item in batch:
                    waitlist.remove(item)
                batch = self.preprocess(batch)
                for item in batch:
                    self.waitlists[item.state].append(item)

            elif i == 1:
                for j,worker in enumerate(self.sampling_status):
                    if len(waitlist) == 0:
                        continue
                    if worker == None:
                        batch = self.select(waitlist,batch_size)
                        for item in batch:
                            waitlist.remove(item)
                        print('Trigger sampling '+ str(len(batch)))
                        
                        parent_conn, child_conn = mp.Pipe()
                        event = mp.Event()
                        process = mp.Process(target=self.iteration, args=(child_conn, j, event),daemon=True)
                        process.start()
                        parent_conn.send(batch)
                        self.sampling_status[j] = (event, parent_conn)
                        
            elif i == 2:
                for j,worker in enumerate(self.decoding_status):
                    if len(waitlist) == 0:
                        continue
                    if worker == None:
                        batch_size = self.batch_configs[i]
                        batch = self.select(waitlist,batch_size)
                        for item in batch:
                            waitlist.remove(item)

                        parent_conn, child_conn = mp.Pipe()
                        event = mp.Event()
                        process = mp.Process(target=self.postprocess, args=(child_conn, j, event),daemon=True)
                        process.daemon = True
                        process.start()
                        parent_conn.send(batch)
                        self.decoding_status[j] = (event, parent_conn)
            
        for i,worker in enumerate(self.sampling_status):
            if worker != None and worker[0].is_set():
                batch = worker[1].recv()
                print(batch)
                for item in batch:
                    if isinstance(item.state,int):
                        self.waitlists[item.state].append(item)
                    else:
                        data_log.append(item.time_list)
                        del item
                self.sampling_status[i] = None
                print('Sampling worker',i,'finished')
        
        for i,worker in enumerate(self.decoding_status):
            if worker != None and worker[0].is_set():
                batch = worker[1].recv()
                for item in batch:
                    if isinstance(item.state,int):
                        self.waitlists[item.state].append(item)
                    else:
                        data_log.append(item.time_list)
                        del item
                self.decoding_status[i] = None
                print('Decoding worker',i,'finished')

    def iteration(self, conn, model_idx, event, **kwargs):
        sampling = conn.recv()
        model,denoiser = getattr(self,'sampling_model_'+str(model_idx))
        sampler = self.state['img_sampler'] if hasattr(sampling[0],'img') else self.state['sampler']
        T = self.state.get('T')
        device = torch.device('cuda:'+str(model_idx))
        with torch.no_grad():
            with autocast('cuda'):
                #with model.ema_scope():
                    if isinstance(sampler, EDMSampler):
                        begin = []
                        for i in sampling:
                            gamma = min(sampler.s_churn / (i.sampling['num_sigmas'] - 1), 2**0.5 - 1) if sampler.s_tmin <= i.sampling['sigmas'][i.sampling['step']] <= sampler.s_tmax else 0.0
                            sigma = i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']]
                            sigma_hat = sigma * (gamma + 1.0)
                            if gamma > 0:
                                eps = torch.randn_like(i.sampling['pic']) * sampler.s_noise
                                i.sampling['pic'] = i.sampling['pic'] + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5            
                            begin.append(sigma_hat)
                        begin = torch.cat(begin,dim=0).to(device)
                    else: 
                        begin = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']] for i in sampling], dim=0).to(device)
                                
                    x = torch.cat([i.sampling['pic'] for i in sampling], dim=0).to(device)
                    end = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step'] + 1] for i in sampling], dim=0).to(device)
                    
                    dict_list = [d.sampling['c'] for d in sampling]
                    cond = {}
                    for key in dict_list[0]:
                        cond[key] = torch.cat([d[key] for d in dict_list], dim=0).to(device)

                    dict_list = [d.sampling['uc'] for d in sampling]
                    uc = {}
                    for key in dict_list[0]:
                        uc[key] = torch.cat([d[key] for d in dict_list], dim=0).to(device)

                    dict_list = [d.sampling['ami'] for d in sampling] 
                    additional_model_inputs = {}
                    for key in dict_list[0]:
                        if key == "image_only_indicator":
                            additional_model_inputs[key] = torch.cat([d[key] for d in dict_list], dim=0).to(device)
                        elif key == "num_video_frames":
                            additional_model_inputs[key] = T
                    lora_dicts = []
                    for d in sampling:
                        for i in range(d.num):
                            lora_dicts.append(d.lora_dict)  
                    lora_dicts = lora_dicts * 2
                    additional_model_inputs['lora_dicts'] = lora_dicts

                    def f_denoiser(input, sigma, c):
                        return denoiser(model, input, sigma, c, **additional_model_inputs)
                    samples = sampler.sampler_step_g(begin,end,f_denoiser,x,cond,uc) if isinstance(sampler, EDMSampler) else sampler.sampler_step(begin,end,denoiser,x,cond,uc)
                    
                    t = 0
                    for i in sampling:
                        i.sampling['pic'] = samples[t:t+i.num].cpu()
                        #print('Finish step ',i.sampling['step'], i.id)
                        i.sampling['step'] = i.sampling['step'] + 1
                        t = t+i.num
                        if i.sampling['step'] >= i.sampling['num_sigmas'] - 1:
                            #print('Finish sampling',i.id)
                            i.sample_z = i.sampling['pic']
                            i.state = 2
        conn.send(sampling)
        print(sampling)
        event.set()
        return
    
    def preprocess(self, encode_process, **kwargs):
        is_image = hasattr(encode_process[0],'img')
        sampler = self.state['img_sampler'] if is_image else self.state['sampler']
        options = self.state.get('options')
        T = self.state.get('T')
        muti_input = self.state.get('muti_input', False)
        with torch.no_grad():
            with autocast("cuda"):
                #with model.ema_scope():
                    value_dicts = [i.value_dict for i in encode_process]
                    batch2model_input = ["num_video_frames", "image_only_indicator"] if T else []
                    if is_image:
                        imgs = []
                        for req in encode_process:
                            imgs += [req.img]*req.num
                        imgs = torch.cat(imgs, dim=0)
                        if self.dtype_half: 
                            imgs = imgs.half()
                    z, c, uc, additional_model_inputs= get_condition(
                        self.state,
                        value_dicts, 
                        sampler=sampler,
                        T=T,
                        batch2model_input=batch2model_input,
                        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
                        muti_input=muti_input,
                        imgs=imgs if is_image else None,
                        lowvram_mode=False,
                        )
                    
                    t = 0
                    t2 = 0
                    for i in encode_process:
                        pic = z[t:t+i.num,].cpu()
                        ic = {k: c[k][t:t+i.num,].cpu() for k in c}
                        iuc = {k: uc[k][t:t+i.num,].cpu() for k in uc}
                        ami = {}
                        for k in additional_model_inputs:
                            if k == "image_only_indicator":
                                ami[k] = additional_model_inputs[k][t2:t2+i.num_samples * 2,].cpu()
                            elif k == "num_video_frames":
                                ami[k] = i.value_dict['T']

                        pic, s_in, sigmas, num_sigmas, ic, iuc = sampler.prepare_sampling_loop(x=pic, cond=ic, uc=iuc, num_steps=i.steps)
                        i.sampling = {
                            'pic':pic, 
                            'step':0,
                            's_in':s_in,
                            'sigmas':sigmas,
                            'num_sigmas':num_sigmas,
                            'c':ic,
                            'uc':iuc,
                            'ami':ami,
                            }
                        i.state = 1
                        t = t + i.num
                        t2 = t2 + i.num_samples * 2
        return encode_process

    def postprocess(self, conn, model_idx, event, **kwargs):
        decode_process = conn.recv()
        model = getattr(self,'decoding_model_'+str(model_idx))
        filter = self.state.get('filter', None)
        return_latents = kwargs.get('return_latents', False)
        scale_factor = kwargs.get('scale_factor', 0.13025)
        device = model.first_stage_model.device

        samples_z = torch.cat([req.sampling['pic'] for req in decode_process], dim=0).to(model.first_stage_model.dtype).to(device)
        with torch.no_grad():
            with autocast('cuda'):
                #with model.ema_scope():
                    z = 1.0 / scale_factor * samples_z
                    if isinstance(model, VideoDecoder):
                        args = {"timesteps": len(z)}
                    else:
                        args = {}
                    out = model.decode(z,**args)
                    samples = torch.clamp((out + 1.0) / 2.0, min=0.0, max=1.0)

                    if filter is not None:
                        samples = filter(samples)

                    if return_latents:
                        return samples, samples_z
                    else:
                        t = 0
                        for req in decode_process:
                            if self.state['T']:
                                output = samples[t:t+req.num]
                                save_video_as_grid_and_mp4(output, req.output_path, self.state['T'], self.state['saving_fps'])
                            else:
                                output = samples[t:t+req.num]
                            t = t + req.num
                            req.state = None
                            req.time_list.append(time.time())
        torch.cuda.empty_cache()
        conn.send(decode_process)
        event = event.set()
        return decode_process

    def create_input(self,num,is_image=False):
        requests = []
        for i in range(num):
            choice = random.choice(sentences)
            req = sd_request(
                        state=optimizer.state,
                        prompt=choice,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        img_path='inputs/00.jpg' if is_image else None,
                        num_samples=1,
                    )
            requests.append(req)
        self.waitlists[0] = self.waitlists[0] + requests

if __name__ == '__main__':
    optimizer = sd_optimizer_muti('configs/sd_xl.yaml')

    mode = 'test'
    if mode == 'server':
        def get_usr_input():
            while True:
                usr_input = input()
                if usr_input != '\n':
                    req = sd_request(
                        state=optimizer.state,
                        prompt=usr_input,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/00.jpg',
                    )
                    optimizer.wait_preprocess.append(req) 

        import threading
        t = threading.Thread(target=get_usr_input)
        t.daemon = True
        t.start()
    elif mode == 'test':
        test_size=512
        time_generate = False
        from datasets import load_dataset
        dataset = load_dataset('lambdalabs/pokemon-blip-captions')
        imgs = dataset['train']['image']
        sentences = dataset['train']['text']
        data_log = []
        #optimizer.model_latency_test(max_num=32,sentences=sentences, imgs=imgs, is_image=True)
        if time_generate:
            sleep_time = 1
            input_num = 1
            freq = input_num/sleep_time
            def timely_generate():
                while True:
                    optimizer.create_input(input_num)
                    time.sleep(sleep_time)
            import threading
            t = threading.Thread(target=timely_generate)
            t.daemon = True
            t.start()
        else:
            optimizer.create_input(test_size,is_image=False)

    while True:
        if mode == 'test':
            if len(data_log) >= test_size:
                break
        optimizer.runtime()

    file_name = f'data_log_muti.json'
    with open(file_name,'w') as f:
        import json
        data = {
                'batch_configs':optimizer.batch_configs,
                'test_size':test_size,
                #'avg':sum(data_log)/len(data_log),
                'dtype': 'half' if optimizer.dtype_half else 'float',
                'generate_speed': freq if time_generate else None,
                'data':data_log,
        }
        json.dump(data,f,indent=4)