from PIL import Image
from IPython.display import display
import torch as th
import sys

import pickle

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# multi GPU implementation of generating synthetic images
num_gpu = 8

def main(id, text_path, save_path):

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)

    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    def save_images_multi(batch: th.Tensor, save_path=['1.png', '2.png']):
        """ Display a batch of images inline. """
        scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu() # B 3 H W
        # reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
        reshaped = scaled.permute(0, 2,3, 1) # B H W 3
        # display(Image.fromarray(reshaped.numpy()))
        # display(Image.fromarray(reshaped.numpy()).save(save_path))
        for i,im in enumerate(reshaped):
            Image.fromarray(im.numpy()).save(save_path[i])

    batch_size = 10
    batch_size_time = 1
    number_each_class = 200
    guidance_scale = 3.0
    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    with open(text_path, 'rb') as f:
        loaded_dict = pickle.load(f)

    classnames = loaded_dict.keys()

    prompt_list = []

    for k,v in loaded_dict.items():
        for sentence in v:
            # prompt_list.append([k,sentence])
            prompt_list.append([k,'a centered satellite photo of '+sentence])
            # prompt_list.append([k,'a photo of '+sentence+', a type of bird'])
            # prompt_list.append([k,'sketch of '+sentence])

    total_len = len(prompt_list)

    if total_len % num_gpu == 0:
        each_len = total_len // num_gpu
    else:
        each_len = total_len // num_gpu +1

    if id != num_gpu-1:
        prompt_list = prompt_list[id*each_len:(id+1)*each_len]
        print('{}:{}'.format(id*each_len,(id+1)*each_len))
    else:
        prompt_list = prompt_list[id * each_len:]
        print('{}:'.format(id * each_len))


    def text2image(prompt, batch_size):


        ##############################
        # Sample from the base model #
        ##############################

        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model.del_cache()

        ##############################
        # Upsample the 64x64 samples #
        ##############################

        tokens = model_up.tokenizer.encode(prompt)
        tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
            tokens, options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model_up.del_cache()
        up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model_up.del_cache()

        return up_samples

    i=0

    os.system('mkdir -p ' + save_path +'/train/')  # ignore_security_alert
    for cls in classnames:
        cls = cls.replace(' ', '_').replace('(', '_').replace(')', '_').replace('/', '_').replace('\'', '')
        dir_name = save_path+'/train/'+cls
        try:
            os.mkdir(dir_name)
        except OSError as error:
            print(error)

    for (classname, prompt) in prompt_list:
        i+=1
        classname = classname.replace(' ','_').replace('(', '_').replace(')', '_').replace('/', '_').replace('\'', '')
        for b in range(batch_size_time):
            up_samples = text2image(prompt, batch_size)
            save_paths = [save_path+'/train/'+classname+'/' + prompt.replace(' ', '_').replace('/', '_').replace('\"', '').replace('\'', '').replace(';', '').replace('<', '').replace('>', '').replace('&', '') + '_'+str(b)+'_' + str(i) + '.png' for i in range(batch_size)]
            save_images_multi(up_samples, save_path=save_paths)


if __name__ == "__main__":
    import os
    import sys
    id = sys.argv[1]
    text_path = sys.argv[2]
    save_path = sys.argv[3]
    print(sys.argv[1])
    id = int(id)

    main(id, text_path, save_path)







