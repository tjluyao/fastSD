from llama import Llama
import time, torch

n_workers = 1
max_bs = 4
n_slots = 102400

class req(object):
    def __init__(self,input,cache_size=None) -> None:
        self.input = input
        self.time = time.time()
        self.max_tokens = 128
        self.state = 0   # 0 refers to INITIATION
        self.buffer = []
        if cache_size:
            self.cache_k = torch.zeros(cache_size)
            self.cache_v = torch.zeros(cache_size)

def Select(pool, n_rsrv):
    batch = []
    pool = sorted(pool, key=lambda req: req.time)

    for item in pool:
        if len(batch) == max_bs:
            break

        if item.state == 0:
            new_n_rsrv = n_rsrv + item.max_tokens
            if new_n_rsrv > n_slots:
                break
            n_rsrv = new_n_rsrv
        
        batch.append(item)

    return batch, n_rsrv

def init_llm(
        ckpt_dir: str = 'Llama-2-7b/',
        tokenizer_path: str = 'Llama-2-7b/tokenizer.model',
        max_seq_len: int = 128,
        max_batch_size: int = 4):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator
    
import random

def generate_random_sentence():
    subjects = ["I", "You", "He", "She", "They"]
    verbs = ["run", "eat", "sleep", "study", "play"]
    objects = ["football", "music", "the piano", "a book", "games"]

    subject = random.choice(subjects)
    verb = random.choice(verbs)
    obj = random.choice(objects)

    sentence = f"{subject} {verb} {obj},"
    return sentence

def random_choose_sentence():
    sentences = [
    "Despite facing numerous challenges, she persevered and eventually achieved her goal of becoming a successful entrepreneur.",
    "The scientific community has made significant strides in understanding the impact of climate change on marine ecosystems.",
    "After years of dedicated practice and training, he finally mastered the art of playing the violin with exceptional skill and precision.",
    "The novel offers a poignant portrayal of the human condition, exploring themes of love, loss, and redemption.",
    "Over the course of history, societies have grappled with questions of morality, justice, and the nature of human existence.",
    "With its diverse landscape and rich cultural heritage, the country has long been a popular destination for tourists from around the world.",
    "The documentary sheds light on the lives of indigenous communities and their struggle to preserve traditional customs in the face of modernization.",
    "Through collaboration and innovation, the research team developed a groundbreaking treatment for a rare genetic disorder.",
    "Despite initial skepticism, the theory gained widespread acceptance within the scientific community and revolutionized our understanding of the universe.",
    "The symphony's haunting melody evokes a sense of nostalgia and longing, resonating deeply with audiences of all ages."
]
    return random.choice(sentences)

if __name__ == '__main__':
    request_pool = []
    n_scheduled = 0
    n_rsrv = 0
    generator = init_llm()
    

    debug = False
    use_cache = True
    if use_cache:
        from llama.model import ModelArgs
        import fairscale.nn.model_parallel.initialize as fs_init
        cache_size = (ModelArgs.n_layers ,1, ModelArgs.max_seq_len, ModelArgs.n_heads // fs_init.get_model_parallel_world_size(), ModelArgs.dim // ModelArgs.n_heads)
    else:
        cache_size = None

    while not debug:
        # Stimulate user input
        if int(time.time()) % 5 == 0 and len(request_pool)<3:
            #text = generate_random_sentence()
            text = random_choose_sentence()
 
            request = req(text,cache_size)
            request.buffer = generator.tokenizer.encode(text, bos=True, eos=False)

            request_pool.append(request)
            #print('Added!',request.time,id(request.cache_k))
        r_batch = []
        batch,n_rsrv = Select(request_pool,n_rsrv)  #batch on req
        if batch and not n_scheduled:
            #print('Select',batch)
            for item in batch:
                #item.state = 1 # Set to RUNNING
                request_pool.remove(item)
            
            r_batch = generator.generate_iter_cache(batch) if use_cache else generator.generate_iter(batch)
            n_scheduled = n_scheduled + 1

        if r_batch:
            #print('Receive',r_batch)
            for item in r_batch:
                if item.state == 3:
                    n_rsrv = n_rsrv-item.max_tokens
                    print('Finish!',item.time,id(item.cache_k))
                    #print(item.buffer)
                    print(generator.tokenizer.decode(item.buffer)+'\n')
                    del item
                else:
                    item.state = 2  # 2 refers to INCREMENT
                    request_pool.append(item)
                    #print(len(item.buffer))
            n_scheduled = n_scheduled - 1 
    else:
        text = "I like playing video games"
        text2 = 'I miss you so much, that is the reason why'
        print(text,text2)
        request = req(text,cache_size)
        request.buffer = generator.tokenizer.encode(text, bos=True, eos=False)
        request2 = req(text,cache_size)
        request2.buffer = generator.tokenizer.encode(text2, bos=True, eos=False)
        while request.state != 3:
            request.state = 2
            generator.generate_iter_cache([request,request2])
        else:
            print(generator.tokenizer.decode(request.buffer)+'\n')
            print(generator.tokenizer.decode(request2.buffer)+'\n')
