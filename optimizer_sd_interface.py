from optimizer_sd import sd_optimizer, sd_request
from einops import rearrange
import streamlit as st

class sd_interface(sd_optimizer):
    def __init__(self, config_file):
        super().__init__(config_file)
        self.display = []

    def runtime(self):
        while True:
            for i,waitlist in enumerate(self.waitlists):
                if len(waitlist) == 0:
                    continue

                batch_size = self.batch_configs[i]
                batch = self.select(waitlist,batch_size)
                for item in batch:
                    waitlist.remove(item)
                
                if i == 0:
                    self.preprocess(batch)
                elif i == 1:
                    self.iteration(batch)
                elif i == 2:
                    self.postprocess(batch)

                for item in batch:
                    if isinstance(item.state,int):
                        self.waitlists[item.state].append(item)
                    else:
                        img = 255.0 * rearrange(item.output.cpu().numpy(), "c h w -> h w c")
                        pass

    def respond(self, text):
        req = sd_request(
                        state=self.state,
                        prompt=text,
                        #lora_pth='lora_weights/EnvySpeedPaintXL01v11.safetensors',
                        video_task=False,
                        #img_path='inputs/00.jpg' if is_image else None,
                        num_samples=1,
                    )
        self.waitlists[0].append(req)

if __name__ == '__main__':
    st.title('Optimizer')
    st.write('This is the optimizer interface')

    