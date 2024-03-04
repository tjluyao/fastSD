from optimizer import Optimizer, yaml, seed_everything
from sd_helper import get_condition, load_model, unload_model, save_video_as_grid_and_mp4, torch, EDMSampler, autocast, perform_save_locally, append_dims, rearrange, image_to_video
from sd_optimizer import sd_request
import random, time
from torchvision.utils import make_grid

class sd_optimizer(Optimizer):
    def __init__(self, config_file):
        config = yaml.load(
            stream=open(config_file, 'r'),
            Loader=yaml.FullLoader
            )
        self.config = config
        self.waitlists=[[],[],[]]
        self.batch_configs = config['batch_configs']
        seed = config.get('seed',49)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seed_everything(seed)

        self.model_name = config['model_name']
        model_config = config['model']
        from sd_helper import init_model as init_sd
        from sd_helper import init_sampling
        self.lowvram_mode = config.get('lowvram_mode',False)
        self.dtype_half = True if config.get('dtype', None) == 'half' else False
        print('Loading model'+' in lowvram'if self.lowvram_mode else '')
        state = init_sd(model_config, load_filter=True, dtype_half=self.dtype_half)
        steps = config.get('default_steps',10)
        state['W'] = model_config.get('W', 1024)
        state['H'] = model_config.get('H', 1024)
        state['C'] = model_config.get('C', 4)
        state['F'] = model_config.get('F', 8)
        state['T'] = model_config.get('T', 6) if self.model_name in ['svd'] else None
        state['options'] = model_config.get('options', {})
        state['options']["num_frames"] = state['T']
        sampler = init_sampling(options=state['options'],steps=steps)
        state['sampler'] = sampler
        state['img_sampler'] = init_sampling(options=state['options'],
                                    img2img_strength=0.75,
                                    steps=steps) if state['T'] is None else None
        state['saving_fps'] = 6
        self.state = state
        self.locations = config.get('locations', ['cuda:0','cuda:0','cuda:1'])
        if not self.lowvram_mode:
            self.load_all(locations= self.locations)
        print('Model loaded')
    
    def load_all(self, locations=['cuda:0','cuda:0','cuda:0']):
        load_model(self.state['model'].conditioner,location=locations[0])
        load_model(self.state['model'].model,location=locations[1])
        load_model(self.state['model'].denoiser, location=locations[1])
        load_model(self.state['model'].first_stage_model, location=locations[2])

    def runtime(self):
        for i,waitlist in enumerate(self.waitlists):
            if len(waitlist) == 0:
                continue

            batch_size = self.batch_configs[i]
            batch = self.select(waitlist,batch_size)
            for item in batch:
                waitlist.remove(item)
            
            if i == 0:
                t_start = time.time()
                self.preprocess(batch)
                t_end = time.time()
            elif i == 1:
                t_start = time.time()
                self.iteration(batch)
                t_end = time.time()
            elif i == 2:
                t_start = time.time()
                self.postprocess(batch)
                t_end = time.time()

            for item in batch:
                item.time_list = item.time_list + [t_start, t_end]
                if isinstance(item.state,int):
                    self.waitlists[item.state].append(item)
                else:
                    del item

    def iteration(self, sampling, **kwargs):
        model = self.state['model']
        sampler = self.state['img_sampler'] if hasattr(sampling[0],'img') else self.state['sampler']
        T = self.state.get('T')
        if self.lowvram_mode:
            load_model(model.model)
            load_model(model.denoiser)
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
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
                        begin = torch.cat(begin,dim=0).to(self.locations[1])
                    else: 
                        begin = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']] for i in sampling], dim=0).to(self.locations[1])
                                
                    x = torch.cat([i.sampling['pic'] for i in sampling], dim=0).to(self.locations[1])
                    end = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step'] + 1] for i in sampling], dim=0).to(self.locations[1])

                    dict_list = [d.sampling['c'] for d in sampling]
                    cond = {}
                    for key in dict_list[0]:
                        cond[key] = torch.cat([d[key] for d in dict_list], dim=0).to(self.locations[1])

                    dict_list = [d.sampling['uc'] for d in sampling]
                    uc = {}
                    for key in dict_list[0]:
                        uc[key] = torch.cat([d[key] for d in dict_list], dim=0).to(self.locations[1])

                    dict_list = [d.sampling['ami'] for d in sampling] 
                    additional_model_inputs = {}
                    for key in dict_list[0]:
                        if key == "image_only_indicator":
                            additional_model_inputs[key] = torch.cat([d[key] for d in dict_list], dim=0).to(self.locations[1])
                        elif key == "num_video_frames":
                            additional_model_inputs[key] = T
                    lora_dicts = []
                    for d in sampling:
                        for i in range(d.num):
                            lora_dicts.append(d.lora_dict)  
                    lora_dicts = lora_dicts * 2
                    additional_model_inputs['lora_dicts'] = lora_dicts

                    def denoiser(input, sigma, c):
                        return self.state["model"].denoiser(self.state["model"].model, input, sigma, c, **additional_model_inputs)
                    samples = sampler.sampler_step_g(begin,end,denoiser,x,cond,uc) if isinstance(sampler, EDMSampler) else sampler.sampler_step(begin,end,denoiser,x,cond,uc)
                    
                    t = 0
                    for i in sampling:
                        i.sampling['pic'] = samples[t:t+i.num]
                        #print('Finish step ',i.sampling['step'], i.id)
                        i.sampling['step'] = i.sampling['step'] + 1
                        t = t+i.num
                        if i.sampling['step'] >= i.sampling['num_sigmas'] - 1:
                            print('Finish sampling',i.id)
                            i.sample_z = i.sampling['pic']
                            i.state = 2
        if self.lowvram_mode:
            unload_model(model.model)
            unload_model(model.denoiser)
        return sampling
    
    def preprocess(self, encode_process, **kwargs):
        state = self.state
        model = state.get('model')
        is_image = hasattr(encode_process[0],'img')
        sampler = state['img_sampler'] if is_image else state['sampler']
        options = state.get('options')
        T = state.get('T')
        muti_input = state.get('muti_input', True)
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
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
                        state,
                        value_dicts, 
                        sampler=sampler,
                        T=T,
                        batch2model_input=batch2model_input,
                        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
                        muti_input=muti_input,
                        imgs=imgs if is_image else None,
                        lowvram_mode=self.lowvram_mode,
                        )
                    
                    t = 0
                    t2 = 0
                    for i in encode_process:
                        pic = z[t:t+i.num,]
                        ic = {k: c[k][t:t+i.num,] for k in c}
                        iuc = {k: uc[k][t:t+i.num,] for k in uc}
                        ami = {}
                        for k in additional_model_inputs:
                            if k == "image_only_indicator":
                                ami[k] = additional_model_inputs[k][t2:t2+i.num_samples * 2,]
                            elif k == "num_video_frames":
                                ami[k] = i.value_dict['T']

                        pic, s_in, sigmas, num_sigmas, ic, iuc = sampler.prepare_sampling_loop(x=pic, cond=ic, uc=iuc, num_steps=i.steps)
                        i.sampling = {'pic':pic, 
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
    
    def postprocess(self,decode_process,**kwargs):
        state = self.state
        model = state.get('model')
        filter = state.get('filter', None)
        return_latents = kwargs.get('return_latents', False)
        samples_z = torch.cat([req.sampling['pic'] for req in decode_process], dim=0).to(model.first_stage_model.dtype).to(model.first_stage_model.device)
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    if self.lowvram_mode:
                        load_model(model.first_stage_model)
                    model.en_and_decode_n_samples_a_time = len(decode_process)
                    samples_x = model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    if self.lowvram_mode:
                        unload_model(model.first_stage_model)

                    if filter is not None:
                        samples = filter(samples)

        if return_latents:
            return samples, samples_z
        else:
            t = 0
            for req in decode_process:
                if self.state['T']:
                    output = samples[t:t+req.num]
                    #img = make_grid(output, nrow=5)
                    #req.output = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    req.output = image_to_video(output, self.state['saving_fps'])
                    #save_video_as_grid_and_mp4(output, req.output_path, self.state['T'], self.state['saving_fps'])
                else:
                    output = samples[t:t+req.num]
                    req.output = 255.0 * rearrange(output[0].cpu().numpy(), "c h w -> h w c")
                    perform_save_locally(req.output_path, output)
                t = t + req.num
                req.state = None

    def fill_input(self):
        if len(self.waitlists[0]) < self.batch_configs[0]:
            choice = random.choice(sentences)
            req = sd_request(
                        state=optimizer.state,
                        prompt=choice,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        num_samples=1,
                    )
            self.waitlists[0].append(req)

    def create_input(self,num,is_image=False):
        requests = []
        for i in range(num):
            choice = random.choice(sentences)
            req = sd_request(
                        state=optimizer.state,
                        prompt=choice,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        image='inputs/00.jpg' if is_image else None,
                        num_samples=1,
                    )
            requests.append(req)
        self.waitlists[0] = self.waitlists[0] + requests

    def model_latency_test(self, max_num, sentences, imgs=None, is_image=False):
        output = {}
        for i in range(1,max_num+1):
            batch = []
            for j in range(i):
                choice = random.choice(sentences)
                im = random.choice(imgs) if is_image else None
                req = sd_request(
                    state=optimizer.state,
                    prompt=choice,
                    num_samples=1,
                    image=im
                    )
                batch.append(req)
            s_time = time.time()
            batch = self.preprocess(batch)
            t1 = time.time() - s_time

            s_time = time.time()
            batch = self.iteration(batch)
            t2 = time.time() - s_time

            s_time = time.time()
            self.postprocess(batch)
            t3 = time.time() - s_time

            output[str(i)] = [t1,t2,t3]
            torch.cuda.empty_cache()
            print(i,t1,t2,t3)
        with open('model_latency.json','w') as f:
            import json
            json.dump(output,f,indent=4)
        exit()

    
if __name__ == '__main__':
    optimizer = sd_optimizer('configs/svd.yaml')

    mode = 'server'
    if mode == 'server':
        def get_usr_input():
            while True:
                usr_input = input()
                if usr_input != '\n':
                    req = sd_request(
                        state=optimizer.state,
                        prompt=usr_input,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=True,
                        image='inputs/00.jpg',
                        num_samples=1,
                    )
                    optimizer.waitlists[0].append(req) 

        import threading
        t = threading.Thread(target=get_usr_input)
        t.daemon = True
        t.start()
    elif mode == 'test':
        time_generate = True
        from datasets import load_dataset
        dataset = load_dataset('lambdalabs/pokemon-blip-captions')
        imgs = dataset['train']['image']
        sentences = dataset['train']['text']
        data_log = []
        #optimizer.model_latency_test(max_num=32,sentences=sentences, imgs=imgs, is_image=True)

        if time_generate:
            generated = 0
            sleep_time = 1
            input_num = 1
            freq = input_num/sleep_time
            def timely_generate():
                global generated
                while True:
                    flag = random.randint(1,100)
                    l = generated
                    bar = 20 + int(l*100/256) if len(data_log) <=256 else 120 - int((l-256)*100/256) 
                    if bar > 100 and flag < bar-100:
                        input_num = 2
                    elif flag < bar:
                        input_num = 1
                    else:
                        input_num = 0
                    optimizer.create_input(input_num)
                    generated = generated + input_num
                    time.sleep(sleep_time)
            import threading
            t = threading.Thread(target=timely_generate)
            t.daemon = True
            t.start()
        else:
            optimizer.create_input(256)

    test_size=512
    while True:
        if mode == 'test':
            #optimizer.fill_input()
            if len(data_log) >= test_size:
                break
        optimizer.runtime()

    file_name = f'data_log_{optimizer.batch_configs}.json'
    with open(file_name,'w') as f:
        import json
        data = {
                'batch_configs':optimizer.batch_configs,
                'test_size':test_size,
                'rolling': True,
                #'avg':sum(data_log)/len(data_log),
                'dtype': 'half' if optimizer.dtype_half else 'float',
                'lowvram_mode': optimizer.lowvram_mode,
                'generate_speed': freq if time_generate else None,
                'data':data_log,
        }
        json.dump(data,f,indent=4)