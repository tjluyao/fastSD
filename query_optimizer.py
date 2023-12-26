import threading
import torch
from pytorch_lightning import seed_everything
from req_sd import init_model, init_sampling, sd_request, VERSION2SPECS, load_model, unload_model, perform_save_locally, save_video_as_grid_and_mp4, get_condition, append_dims, autocast
from sgm.modules.diffusionmodules.sampling import EDMSampler
from req_llama import llama_request, hf_llama_request, get_cache_size
from req_llava import init_llava, llava_req
from req_llamalora import init_Llama_lora, llama_lora_req, init_lora
from punica import BatchLenInfo, BatchedKvCache, BatchedLlamaLoraWeight, KvPool

class query_optimazer:
    def __init__(self,
                 model_name: str = 'lora',
                 output_path: str = 'outputs/',
                 seed: int = 49,
                 **kwargs
                 ):
        self.model_name = model_name
        self.model = self.init_model(model_name,**kwargs)
        
        self.output_path = output_path
        self.max_batch_size = 4
        self.n_slots = 102400

        self.n_rsrv = 0
        self.n_scheduled = 0
        self.wait_runtime = []

        self.batch_option = 1
        self.wait_preprocess = []
        self.wait_postprocess = []
        seed_everything(seed)
        pass

    def init_model(self,model_name,**kwargs):
        if model_name == 'llama':
            '''
            from llama import Llama
            generator = Llama.build(
                ckpt_dir='Llama-2-7b/',
                tokenizer_path='Llama-2-7b/tokenizer.model',
                max_seq_len=128,
                max_batch_size=4,
            )
            '''
            from llama_generator import Large_model
            generator = Large_model()
            print('Model loaded')
            return generator
        
        elif model_name == 'hf_llama':
            from transformers import LlamaForCausalLM, AutoTokenizer
            model = LlamaForCausalLM.from_pretrained('./checkpoints/llava-v1.5-7b')
            tokenizer = AutoTokenizer.from_pretrained('./checkpoints/llava-v1.5-7b')
            self.tokenizer = tokenizer
            return model
        
        elif model_name == 'llava':
            model, self.tokenizer, self.image_processor = init_llava('./checkpoints/llava-v1.5-7b')
            self.device = model.device
            self.kv_pool = KvPool(
                num_layers=model.config.num_hidden_layers,
                num_heads=model.config.num_attention_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
                page_len=16,
                dtype=torch.float16,
                device=model.device,
            )
            return model
        
        elif model_name == 'lora':
            lora_ids = kwargs.get('lora_ids', [])
            model, tokenizer, model_config, kvpool = init_Llama_lora('./checkpoints/llava-v1.5-7b', 'cuda:0')
            self.tokenizer = tokenizer
            self.model_config = model_config
            self.kvpool = kvpool
            self.device = model.device
            self.lora_weights = init_lora(lora_ids, 
                                          model_config, 
                                          device=self.device,
                                          )
            return model
            
        
        elif model_name in VERSION2SPECS:
            version_dict = VERSION2SPECS[model_name]
            state = init_model(version_dict, load_filter=True)
            steps = kwargs.get('steps',10)
            state['W'] = 512
            state['H'] = 512
            state['C'] = version_dict['C']
            state['F'] = version_dict['f']
            state['T'] = 6 if model_name in ['sdv'] else None
            state['options'] = version_dict.get('options', {})
            state['options']["num_frames"] = state['T']
            sampler = init_sampling(options=state['options'],steps=steps)
            state['sampler'] = sampler
            state['saving_fps'] = 6
            self.state = state
            load_model(state['model'].model)
            load_model(state['model'].denoiser)
            print('Model loaded')
            return state.get('model')
        
        else:
            raise NotImplementedError
        
    def add_request(self,req):
        if self.model_name in ['llama','llava','hf_llama','lora']:
            self.wait_runtime.append(req)

        elif self.model_name in VERSION2SPECS:
            self.wait_preprocess.append(req)
        else:
            raise NotImplementedError
    
    def select_batch(self, pool):
        batch = []
        pool = sorted(pool, key=lambda req: req.time)

        for item in pool:
            if len(batch) == self.max_batch_size:
                break

            if item.state == 0:
                if self.model_name == 'llama':
                    new_n_rsrv = self.n_rsrv + item.max_tokens
                    if new_n_rsrv > self.n_slots:
                        break
                    self.n_rsrv = new_n_rsrv
            batch.append(item)
        return batch
    
    def runtime(self, model, **kwargs):
        use_cache = kwargs.get('use_cache',True)
        r_batch = []
        batch = self.select_batch(self.wait_runtime)
        if batch and not self.n_scheduled:
            for item in batch:
                self.wait_runtime.remove(item)
            if self.model_name == 'llama':
                r_batch = model.generate_iter_cache(batch) if use_cache else model.generate_iter(batch)

            elif self.model_name == 'hf_llama':
                r_batch = self.llama_iterate(batch)

            elif self.model_name == 'llava':
                r_batch = self.llava_iterate(batch)

            elif self.model_name == 'lora':
                r_batch = self.lora_iterate(batch)

            elif self.model_name in VERSION2SPECS:
                r_batch = self.sd_sample(batch, self.state)
            else:
                raise NotImplementedError
            self.n_scheduled = self.n_scheduled + 1

        if r_batch:
            for item in r_batch:
                if item.state == 3:
                    if self.model_name == 'llama':
                        self.n_rsrv = self.n_rsrv-item.max_tokens
                        print('Finish!',item.time,id(item.cache_k))
                        print(model.tokenizer.decode(item.buffer)+'\n')
                        del item
                    elif self.model_name == 'hf_llama':
                        print('Finish!',item.time)
                        print(self.tokenizer.decode(item.buffer)+'\n')
                        del item
                    elif self.model_name == 'llava':
                        print('Finish!',item.time)
                        txt = self.tokenizer.decode(
                            item.generator.remove_unvalid(item.generator.output_ids),
                            skip_special_tokens=True,
                            spaces_between_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        )
                        print(txt+'\n')
                        del item
                    elif self.model_name == 'lora':
                        print('Finish!',item.time)
                        print(item.generator.decode_tokens())
                        del item

                    elif self.model_name in VERSION2SPECS:
                        self.wait_postprocess.append(item)
                else:
                    item.state = 2
                    self.wait_runtime.append(item)
            self.n_scheduled = self.n_scheduled - 1 

    def llama_iterate(self, batch):
        for item in batch:
            item.buffer = self.model.generate(
                input_ids=item.buffer,
                use_cache=True,
                past_key_values=item.cache,
                max_length=len(item.buffer[0]) + 1,
            )
            if len(item.buffer[0]) >= item.max_tokens:
                item.state = 3
                item.buffer = item.buffer[:item.max_tokens]
                item.cache = None
        return batch
    
    def llava_iterate(self, batch):
        next_ids = []
        image_tensors = []
        kvcaches = []
        decode_input_ids = []
        for item in batch:
            item.kvcache.acquire_one()
            next_ids.append(item.next_token_id)
            image_tensors.append(item.image_tensor)
            kvcaches.append(item.kvcache)
            decode_input_ids.append(item.generator.output_ids[-1])
        logits, _ = self.model(
            input_ids=torch.tensor(next_ids, dtype=torch.long, device=self.device).unsqueeze(0),
            blen=BatchLenInfo([], len(decode_input_ids), self.device),
            prefill_kv=None,
            decode_kv=BatchedKvCache(kvcaches),
            images=image_tensors,
        )
        for i, item in enumerate(batch):
            item.next_token_id = item.generator.get_next_token_id(logits.squeeze(0)[i].unsqueeze(0))
            item.generator.append_token(item.next_token_id)
            if item.generator.is_stop():
                item.state = 3
                item.cache = None

        return batch
    
    def lora_iterate(self, batch):
        prefill_input_ids, prefill_lens, prefill_kv = [], [], []
        decode_input_ids, decode_kv = [], []
        lora_ids, lora_lens = [], []
        for item in batch:
            reqctx = item.generator
            if reqctx.is_prefill():
                prefill_input_ids.extend(reqctx.output_ids)
                prefill_lens.append(len(reqctx.output_ids))
                prefill_kv.append(reqctx.kvcache)
            else:
                decode_input_ids.append(reqctx.output_ids[-1])
                decode_kv.append(reqctx.kvcache)
                reqctx.kvcache.acquire_one()
            if lora_ids and lora_ids[-1] == reqctx.lora_id:
                lora_lens[-1] += 1
            else:
                lora_ids.append(reqctx.lora_id)
                lora_lens.append(1)
        input_ids = torch.tensor(
                prefill_input_ids + decode_input_ids,
                dtype=torch.long,
                device=self.device,
            )
        blen = BatchLenInfo(prefill_lens, len(decode_input_ids), self.device)
        prefill_kv = BatchedKvCache(prefill_kv) if prefill_kv else None
        decode_kv = BatchedKvCache(decode_kv) if decode_kv else None
        lora = BatchedLlamaLoraWeight(
            [self.lora_weights[id] for id in lora_ids], lora_lens
        )
        logits, _ = self.model(input_ids, blen, prefill_kv, decode_kv, lora)
        if prefill_kv:
            if decode_kv:
                logits = torch.cat(
                    [logits[blen.indptr[1:] - 1], logits[blen.doff :]]
                )
            else:
                logits = logits[blen.indptr[1:] - 1]

        for i, item in enumerate(batch):
            reqctx = item.generator
            next_token_id = reqctx.get_next_token_id(logits[i].unsqueeze(0))
            reqctx.append_token(next_token_id)
            if reqctx.is_stop():
                item.state = 3
                reqctx.kvcache.release()
                reqctx.kvcache = None
        return batch
    
    def check_prepost(self):
        if len(self.wait_preprocess) >= self.batch_option:
            print('Start encoding')
            encode_process = self.wait_preprocess
            self.wait_preprocess = []
            encode_process = self.preprocess(encode_process, self.state)
            for i in encode_process:
                print('Finish encoding',i.id)
                self.wait_runtime.append(i)

        if len(self.wait_postprocess) >= self.batch_option:
            print('Start decoding')
            decode_process = self.wait_postprocess
            self.wait_postprocess = []
            samples_z = torch.cat([i.sampling['pic'] for i in decode_process], dim=0)
            samples = self.postprocess(samples_z, self.state)
            if self.model_name not in ['svd']:
                perform_save_locally(self.output_path, samples)  #Save to local file
            if self.state['T']:
                save_video_as_grid_and_mp4(samples, self.output_path, self.state['T'], self.state['saving_fps'])
            for i in decode_process:
                print('Saved',i.id)
                decode_process.remove(i)
                del i

    def preprocess(self, encode_process,state,muti_input=True):
        model = state.get('model')
        sampler = state.get('sampler')
        options = state.get('options')
        T = state.get('T')
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    value_dicts = [i.value_dict for i in encode_process]
                    batch2model_input = ["num_video_frames", "image_only_indicator"] if T else []
                    z, c, uc, additional_model_inputs= get_condition(
                        state,
                        value_dicts, 
                        sampler=sampler,
                        T=T,
                        batch2model_input=batch2model_input,
                        force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
                        force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
                        muti_input=muti_input,
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
                        t = t + i.num
                        t2 = t2 + i.num_samples * 2
        return encode_process

    def postprocess(self, samples_z, state, return_latents=False):
        model = state.get('model')
        filter = state.get('filter', None)
        with torch.no_grad():
            with autocast("cuda"):
                with model.ema_scope():
                    load_model(model.first_stage_model)
                    model.en_and_decode_n_samples_a_time = 2
                    samples_x = model.decode_first_stage(samples_z)
                    samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    unload_model(model.first_stage_model)

                    if filter is not None:
                        samples = filter(samples)

                    if return_latents:
                        return samples, samples_z
                    return samples
    
    def sd_sample(self, sampling, state):
        model = state.get('model')
        sampler = state.get('sampler')
        T = state.get('T')
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
                        begin = torch.cat(begin,dim=0)
                    else: 
                        begin = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step']] for i in sampling], dim=0)
                                
                    x = torch.cat([i.sampling['pic'] for i in sampling], dim=0)
                    end = torch.cat([i.sampling['s_in'] * i.sampling['sigmas'][i.sampling['step'] + 1] for i in sampling], dim=0)

                    dict_list = [d.sampling['c'] for d in sampling]
                    cond = {}
                    for key in dict_list[0]:
                        cond[key] = torch.cat([d[key] for d in dict_list], dim=0)

                    dict_list = [d.sampling['uc'] for d in sampling]
                    uc = {}
                    for key in dict_list[0]:
                        uc[key] = torch.cat([d[key] for d in dict_list], dim=0)

                    dict_list = [d.sampling['ami'] for d in sampling] 
                    additional_model_inputs = {}
                    for key in dict_list[0]:
                        if key == "image_only_indicator":
                            additional_model_inputs[key] = torch.cat([d[key] for d in dict_list], dim=0)
                        elif key == "num_video_frames":
                            additional_model_inputs[key] = T

                    lora_dicts = []
                    for d in sampling:
                        for i in range(d.num):
                            lora_dicts.append(d.lora_dict)  
                    lora_dicts = lora_dicts * 2
                    additional_model_inputs['lora_dicts'] = lora_dicts

                    def denoiser(input, sigma, c):
                        return state["model"].denoiser(state["model"].model, input, sigma, c, **additional_model_inputs)
                    samples = sampler.sampler_step_g(begin,end,denoiser,x,cond,uc) if isinstance(sampler, EDMSampler) else sampler.sampler_step(begin,end,denoiser,x,cond,uc)
                                        
                    t = 0
                    for i in sampling:
                        i.sampling['pic'] = samples[t:t+i.num]
                        print('Finish step ',i.sampling['step'], i.id)
                        i.sampling['step'] = i.sampling['step'] + 1
                        t = t+i.num
                        if i.sampling['step'] >= i.sampling['num_sigmas'] - 1:
                            print('Finish sampling',i.id)
                            i.sample_z = i.sampling['pic']
                            i.state = 3
                    return sampling

def get_usr_input():
    while True:
        usr_input = input()
        if usr_input != '\n':
            if optimatizer.model_name == 'llama':
                cache_size = get_cache_size(optimatizer.model)
                req = llama_request(usr_input,optimatizer.model,cache_size=cache_size)

            elif optimatizer.model_name == 'hf_llama':
                req = hf_llama_request(usr_input,optimatizer.tokenizer)

            elif optimatizer.model_name == 'llava':
                req = llava_req(usr_input,
                                img_path='inputs/01.jpg',
                                tokenizer=optimatizer.tokenizer,
                                model=optimatizer.model,
                                config=optimatizer.model.config,
                                image_processor=optimatizer.image_processor,
                                kv_pool=optimatizer.kv_pool,
                                )
            elif optimatizer.model_name == 'lora':
                req = llama_lora_req(usr_input,  
                                     kvpool=optimatizer.kvpool, 
                                     tokenizer=optimatizer.tokenizer, 
                                     lora_id='Chinese',
                                     )
                optimatizer.add_request(req)
                req = llama_lora_req(usr_input, 
                                     kvpool=optimatizer.kvpool, 
                                     tokenizer=optimatizer.tokenizer, 
                                     lora_id='fin',
                                     )
            elif optimatizer.model_name in VERSION2SPECS:
                #req = sd_request(state=optimatizer.state,img_path='inputs/'+usr_input+'.jpg')
                req = sd_request(state=optimatizer.state,lora_pth='lora_weights/pixel-art-xl.safetensors',prompt=usr_input)
            else:
                raise NotImplementedError
            print('New request',req.time)
            optimatizer.add_request(req)

if __name__ == "__main__":
    optimatizer = query_optimazer('lora',lora_ids=['fin','Chinese'])
    input_thread = threading.Thread(target=get_usr_input)
    input_thread.daemon = True
    input_thread.start()
    while True:
        optimatizer.check_prepost()
        optimatizer.runtime(optimatizer.model)