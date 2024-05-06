import asyncio
from aiohttp import web
import aiohttp_cors
from io import BytesIO

import accelerate
import safetensors.torch as safetorch
import torch
from tqdm import trange, tqdm

from PIL import Image

import k_diffusion as K


def main():
    checkpoint = './mm1_level.safetensors'
    steps = 50
    size = 64

    config = K.config.load_config(checkpoint)
    model_config = config['model']
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
    inner_model.load_state_dict(safetorch.load_file(checkpoint))

    accelerator.print('Parameters:', K.utils.n_params(inner_model))
    model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    @torch.no_grad()
    @K.utils.eval_mode(model)
    def run() -> Image.Image:
        if accelerator.is_local_main_process:
            tqdm.write('Sampling...')
        sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho=7., device=device)
        def sample_fn(n):
            x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
            x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_local_main_process)
            return x_0
        x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, size, size)
        if accelerator.is_main_process:
            img_size = x_0[0].shape[0]
            full_image = Image.new('RGB', (img_size, size * img_size))
            for i, out in enumerate(x_0):
                base_img = K.utils.to_pil_image(out)
                full_image.paste(base_img, (0, i * img_size))
            return full_image
        return Image.new('RGB', (0, 0))


    routes = web.RouteTableDef()

    @routes.get('/')
    async def hello(request):
        return web.Response(text="Hello, world")

    @routes.get('/generate')
    async def generate(request):
        await asyncio.sleep(3)
        img = run()
        membuf = BytesIO()
        img.save(membuf, format="png") 
        return web.Response(body=membuf.getvalue(), content_type="image/png")
    

    app = web.Application()
    cors = aiohttp_cors.setup(app)

    r = app.add_routes(routes)
    for route in r:
        cors.add(route, {
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    web.run_app(app)


if __name__ == '__main__':
    main()