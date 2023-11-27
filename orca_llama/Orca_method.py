import time
n_workers = 4
max_bs = 8
n_slots = 102400

class req(object):
    def __init__(self,message) -> None:
        self.message = message
        self.time = time.time()
        self.max_tokens = 128
        self.state = 0   # 0 refers to INITIATION

def Select(pool, n_rsrv):
    batch = []
    
    for i in len(pool):
        if pool[i].state != 1 :    # 1 refers to RUNNING
            pool.pop(i)

    pool.sort()
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
        
def wait_for_return(batch):
    return batch

def is_finish(item):
    return 1

def run_batch():
    return
if __name__ == '__main__':
    request_pool = []
    n_scheduled = 0
    n_rsrv = 0
    while True:
        batch,n_rsrv = Select(request_pool,n_rsrv)  #batch on req

        for item in batch:
            item.state = 1
        n_scheduled = n_scheduled + 1
        run_batch(batch)
        if n_scheduled == n_workers:
            r_batch = wait_for_return()
            for item in r_batch:
                item.state = 2  # 2 refers to INCREMENT
                if is_finish(item):
                    n_rsrv = n_rsrv-item.max_tokens
                else:
                    request_pool.append(item)
            n_scheduled = n_scheduled - 1 

