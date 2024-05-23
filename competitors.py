import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import GaussianBlur
import utils
from geex import ExplainerWithBaseline

from skimage.transform import resize

class GradBased:
    def _get_grad(self, m, img, lbl):
        img = img.requires_grad_(True)
        output = m(img)[:, lbl]
        return torch.autograd.grad(output, img)[0].cpu()

class SmoothGrad(GradBased):
    def __init__(self, num=50, sigma=0.75):
        super(SmoothGrad, self).__init__()
        self.num, self.sigma = num, sigma
        self.pv_floor, self.pv_ceil = 0., 1.
    
    def set_pixel_value_bounds(self, pv_floor, pv_ceil):
        self.pv_floor, self.pv_ceil = pv_floor, pv_ceil

    def explain(self, m, img):
        device = utils.get_device(m)
        lbl = np.argmax(m(img.to('cuda'))[0].detach().cpu())
        grad = self._get_grad(m, img.to(device), lbl)[0]
        for _ in range(self.num-1):
            noise = torch.randn_like(img, dtype=img.dtype) / 2
            noised = torch.add(noise * self.sigma, img)
            noised = torch.clamp(noised.permute(0, 2, 3, 1), self.pv_floor, self.pv_ceil)
            noised = noised.permute(0, 3, 1, 2)
            grad += self._get_grad(m, noised.to('cuda'), lbl)[0]
        return grad/self.num
 
class IntegratedGrad(GradBased, ExplainerWithBaseline):
    def __init__(self, steps, input_size, baseline=None):
        super(IntegratedGrad, self).__init__()
        self.baseline = baseline if baseline is not None else torch.zeros(input_size)
        self.steps = steps
        self.smooth_num, self.smooth_sigma = 1, 0.5
        self.aligned_noise = False
        self.blur_m = None
        if isinstance(self.baseline, str) and self.baseline.lower() == 'blur':
            self.set_bluring_kernel()   

    def set_bluring_kernel(self, blur_m=None):
        self.blur_m = blur_m if blur_m is not None else GaussianBlur((31, 31), 5.0)

    def enable_smooth(self, num: int, sigma=0.5, aligned_noise=False):
        self.smooth_num = num
        self.smooth_sigma = sigma
        self.aligned_noise = aligned_noise

    def _get_smoothed_path(self, img, baseline):
        path_samples = self._get_path(img, baseline)
        if self.smooth_num > 1:
            samples = []
            noise = torch.randn((self.smooth_num-1,) + img.shape[1:], dtype=img.dtype) / 2
            for sample in path_samples:
                if not self.aligned_noise:
                    noise = torch.randn((self.smooth_num-1,) + img.shape[1:], dtype=img.dtype) / 2
                noised = torch.add(noise * self.smooth_sigma, sample)
                samples += list(noised.unsqueeze(1))
            path_samples = samples
        return path_samples

    def explain(self, m, img):
        device = utils.get_device(m)
        lbl = np.argmax(m(img.to('cuda'))[0].detach().cpu())
        baseline = self._get_baseline(img)
        grad = torch.zeros(img.shape[-3:])

        samples = self._get_smoothed_path(img, baseline)
        for sample in samples:
            grad += self._get_grad(m, sample.to(device), lbl)[0] 
        
        dx = img - baseline
        grad = grad * dx[0] / self.steps
        return grad 
    
   
class RISE(nn.Module, ExplainerWithBaseline):
    def __init__(self, model, input_size, gpu_batch=100, baseline=0):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.baseline = baseline
        self.blur_m = GaussianBlur((31, 31), 5.0)

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        # np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        baseline = self._get_baseline(x)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            # Apply array of filters to the image
            stack = torch.mul(self.masks[i:min(i + self.gpu_batch, N)].cuda(), x.data)
            stack += torch.mul((1-self.masks[i:min(i + self.gpu_batch, N)]).cuda(), baseline)
            p.append(self.model(stack).detach().cpu())
            # p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal
   
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal