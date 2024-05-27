import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pickle
from torchvision.transforms import GaussianBlur
import utils

class ExplainerWithBaseline(object):
    def __init__(self):
        super().__init__()
        self.available_choices = {
            'interpolate': self._interpolate,
            }
        self.path_mode = 'interpolate'
    
    def _get_baseline(self, x):
        if isinstance(self.baseline, str) and self.baseline.lower() == 'blur':   
            # explicand-specific baseline, requiring definition of 'self.blur_m', only apply to IMG
            if self.blur_m is None: 
                print('Bluring kernel is not defined, use default')
                self.set_bluring_kernel()
            baselines = self.blur_m(x)
        elif isinstance(self.baseline, (int, float)):
            # Constant value baseline, identical value across different channels (if applies)
            baselines = self.baseline
        elif isinstance(self.baseline, torch.Tensor):
            # Image-like baseline, having shape [C*H*W]
            baselines = self.baseline
        elif isinstance(self.baseline, tuple):
            # Constant value baseline for multi-channel image only
            baselines = torch.Tensor(self.baseline).reshape(len(self.baseline),1,1)
        else:
            assert 1 == 0, 'Unsupported baseline type'
        return baselines
    
    def set_path_mode(self, mode: str):
        self.path_mode = mode

    def _interpolate(self, img, baseline, *args):
        return [baseline + (float(i)/self.steps) * (img - baseline) for i in range(self.steps+1)]
    
    def _get_path(self, img, baseline, *args):
        try:
            func = self.available_choices[self.path_mode]
            return func(img, baseline, *args)
        except KeyError:
            assert 1==0, f'Unknown path mode, only support {self.available_choices.keys()}'
 
# Section 3.1 Gradient Estimation
class GE(object):
    def __init__(self, mask_num:int, sigma:float, input_size):
        """
        Parameters:
        -----------
        mask_num:   n^*, number of observations/queries by a gradient estimator
        sigma:      spread of the search distribution (a Gaussian)
        input_size: a tuple/list describing the shape of expected inputs
        mask_size:  a tuple/list describing the initial size of masks,
                    which are later upsampled to the targeted input size.
                    Mask resizing is disabled if the argument is set to None.
        sym_noise:  flag for mirror sampling, enabled by default
        """
        super().__init__()
        self.mask_num, self.sigma = mask_num, sigma
        self.input_size = input_size
        self.fitness_fn = None  # the loss is f(x) for one class, specified by the 'explain()' function
        self.pv_floor, self.pv_ceil = None, None
        self.sym_noise = True

        # Initialization of masks/noises
        self.masks = self._noise_masks((1, *input_size[1:]), mask_num) / 2 

    def set_fitness_fn(self, fitness_fn):
        self.fitness_fn = fitness_fn    

    def explain(self, m, img, batch_size=64, verbose=False, target_class=None):
        """
        Parameters:
        -----------
        m:      target model, expecting the forward function as outcome interface in shape [B*L],
                where B denotes the batch size and L indicates the number of possible classes.
        img:    (torch.Tensor) in shape [B*C*H*W], B=1.
        batch_size:     (int) batch size of queries for acquiring observations f(z)
        target_class:   (int) the index of the class of interest.
                        Focus on the predicted class if not given

        Return:
        -----------
        eta: estimated gradient for each feature, having the same shape of an image [H*W]
        """
        total_n = len(self.masks)
        with torch.no_grad():
            ges = []
            lbl = m(img.to(utils.get_device(m))).detach().cpu().argmax().item() 
            target_class = lbl if target_class is None else target_class
            for i in tqdm(range(0, total_n, batch_size), disable=not verbose):
                ptr = min(i+batch_size, total_n)
                batch_weight = float(ptr - i) / batch_size

                # Applying masks on img for generating queries 'z'
                masks = self.masks[i: ptr] 
                masks = self._get_symmetric_masks(masks)
                queries = self._apply_noises(img, masks)

                # Get observations from the outcome of the target 'm'
                fitness = self._fitness(m, queries, target_class).detach().cpu()
                # fitness = fitness - base if self.var_reduction else fitness

                # Estimating gradient with given observation, collecting the result of current batch
                ge_batch = self._gradient_estimate(fitness, masks)
                ge_batch = torch.mean(ge_batch, dim=0, keepdim=True) * batch_weight
                ges.append(ge_batch)
            # Averaging over batch results 
            return torch.mean(torch.cat(ges), dim=0) 

    def _gradient_estimate(self, fitness, masks):
        """
        Parameters:
        -----------
        fitness:    (float) the observations given by f(z), aligned with masks
        masks:      (torch.Tensor) the masks

        Return:
        -----------
        eta: estimated gradient for each feature, having the same shape of an image [H*W]
        """
        shape = fitness.shape if fitness.ndim == 2 else (fitness.shape[0], 1)
        shape = shape + (1, 1)
        fitness = torch.tile(fitness.reshape(shape), (1,1) + self.input_size[1:])
        return masks * fitness / (self.sigma**2)     # sum(f(z) * (z - mu)/(sigma^2))

    def _apply_noises(self, img, masks, *args):
        noised = torch.add(masks * self.sigma, img)
        if self.pv_floor is not None:
            noised = torch.clamp(noised.permute(0, 2, 3, 1), self.pv_floor, self.pv_ceil)
            noised = noised.permute(0, 3, 1, 2)
        return noised

    def _fitness(self, m, imgs, target=None):
        device = utils.get_device(m)
        logits = m(imgs.to(device))
        if self.fitness_fn:     
            # the loss is the cross entropy comparing current outcome to the non-informative state {1/L} * L
            classes = len(logits[0])
            target = torch.ones_like(logits, device=device) / classes 
            return self.fitness_fn(logits, target)
        else:
            return logits[:, target]

    def mask_smoothing(self, filter_width=5, filter_sigma=0.7, batch_size=64):
        """
        Mask smoothing with a Gaussian filter,
        call only ONCE while initializing the explainer
        """
        self.masks = self._mask_smoothing(self.masks, filter_width, filter_sigma, batch_size)

    def _get_symmetric_masks(self, masks):
        return torch.cat([masks, -masks]) if self.sym_noise else masks

    @staticmethod
    def _mask_smoothing(orgs, filter_width=5, filter_sigma=0.7, batch_size=64):
        if filter_width <= 0:
            return orgs
        smoother = FixedGaussianFilter((filter_width, filter_width), filter_sigma)
        n = len(orgs)
        masks = []
        for i in tqdm(range(0, n, batch_size), desc='Smoothing masks'):
            ptr = i + batch_size
            resized = smoother(orgs[i:ptr]).detach()
            masks.append(resized)
        return torch.cat(masks)

    @staticmethod
    def _noise_masks(shape, num):
        return torch.randn((num, ) + shape, dtype=torch.float32)
        
"""
Section 3.2 GEEX, vanilla implementation with straightforward interpolation
"""
class GEEX_raw(GE, ExplainerWithBaseline):
    def __init__(self, mask_num, sigma, input_size, steps=10, baseline=None):
        """
        Parameters:
        -----------
        steps:      (int) interpolation steps, higher better, but takes also longer
        baseline:   specifying the baseline for the explainer, 
                    refer to '_get_baseline()' for details
        """
        super().__init__(mask_num, sigma, input_size)
        self.baseline = baseline if baseline is not None else torch.zeros(input_size)
        self.steps = steps
        self.blur_m = None

    def set_bluring_kernel(self, blur_m=None):
        """
        Parameters:
        -----------
        blur_m: blurring kernel for acquiring the baseline, 
                used only when specifying self.baseline='blur'
        """
        self.blur_m = blur_m if blur_m is not None else GaussianBlur((31, 31), 5.0)

    def explain(self, m, img, batch_size=64, verbose=False, target_class=None, get_raw=False):
        baseline = self._get_baseline(img) 
        with torch.no_grad():
            # Do interpolation between the baseline and the explicand, then explain each interpolation
            imgs = [baseline + (float(i)/self.steps) * (img - baseline) for i in range(self.steps+1)]
            grad, dx = torch.zeros(img.shape[-3:]), img - baseline
            for sample in imgs:
                grad += GE.explain(self, m, sample, batch_size, verbose, target_class)
            raw = grad.detach().clone()

            # \frac{x - \mathring{x}}{steps} * \sum_i{\eta^{(i)}}
            grad = grad * dx[0] / self.steps

            # return raw only for debugging purpose
            return (grad, raw) if get_raw else grad     
       
"""
Section 3.3 GEEX -- Integrating dense one-sample gradient estimators
"""    
class GEEX(GE, ExplainerWithBaseline):
    def __init__(self, mask_num, sigma, input_size, baseline=None):
        super().__init__(mask_num, sigma, input_size)
        self.baseline = baseline if baseline is not None else torch.zeros(input_size)
        self.alphas = torch.rand(mask_num)  # Uniformly sampled points on path
        self.blur_m = None
        if isinstance(self.baseline, str) and self.baseline.lower() == 'blur':
            # use default bluring kernel for baseline
            # the option can be overwritten by explicitly specifying a kernel with 'set_bluring_kernel'
            self.set_bluring_kernel()   

    def set_bluring_kernel(self, blur_m=None):
        """ Blurring kernel that defines the baseline for every explicand """
        self.blur_m = blur_m if blur_m is not None else GaussianBlur((31, 31), 5.0)

    def _interpolate(self, img, baseline, *args):
        """ Interpolated instances are returned in BATCH, differing from the output in list for IG """
        alphas = args[0]
        return alphas.view(-1,1,1,1) * img + (1-alphas).view(-1,1,1,1) * baseline

    def _get_path_args(self, ptr_l, ptr_r, device):
        if self.path_mode == 'interpolate':
            path_args = self.alphas[ptr_l: ptr_r].to(device)
        else:
            assert 1==0, f'Unknown path mode, only support {self.available_choices.keys()}'
        path_args = torch.cat([path_args, path_args]) if self.sym_noise else path_args
        return path_args

    def explain(self, m, img, batch_size=64, verbose=False, target_class=None, get_raw=False):
        total_n, baseline = len(self.masks), self._get_baseline(img)
        device = utils.get_device(m)
        if isinstance(baseline, torch.Tensor):
            baseline = baseline.to(device)
        with torch.no_grad():
            ges = []
            img = img.to(device)
            lbl = m(img).detach().cpu().argmax().item() 
            target_class = lbl if target_class is None else target_class
            for i in tqdm(range(0, total_n, batch_size), disable=not verbose):
                ptr = min(i+batch_size, total_n)
                batch_weight = float(ptr - i) / batch_size

                # Combining masks and alphas for generating queries
                masks = self._get_symmetric_masks(self.masks[i: ptr].to(device))
                path_args = self._get_path_args(i, ptr, device)

                samples = self._get_path(img, baseline, path_args)
                queries = self._apply_noises(samples, masks, baseline)

                fitness = self._fitness(m, queries, target_class).to(device)

                ge_batch = self._gradient_estimate(fitness, masks)
                ge_batch = torch.mean(ge_batch, dim=0, keepdim=True) * batch_weight
                ges.append(ge_batch)
            raw = torch.mean(torch.cat(ges), dim=0)
            diff = (img - baseline)[0] 
            grad = raw * diff
            return (grad.cpu(), raw.cpu()) if get_raw else grad.cpu()
        
class FixedGaussianFilter:
    def __init__(self, kernel_size, sigma, dtype: torch.dtype=torch.float32, device: torch.device='cpu') -> None:
        if not isinstance(kernel_size, (int, list, tuple)):
            raise TypeError(f"kernel_size should be int or a sequence of integers. Got {type(kernel_size)}")
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if len(kernel_size) != 2:
            raise ValueError(f"If kernel_size is a sequence its length should be 2. Got {len(kernel_size)}")
        for ksize in kernel_size:
            if ksize % 2 == 0 or ksize < 0:
                raise ValueError(f"kernel_size should have odd and positive integers. Got {kernel_size}")

        if sigma is None:
            sigma = [ksize * 0.15 + 0.35 for ksize in kernel_size]

        if sigma is not None and not isinstance(sigma, (int, float, list, tuple)):
            raise TypeError(f"sigma should be either float or sequence of floats. Got {type(sigma)}")
        if isinstance(sigma, (int, float)):
            sigma = [float(sigma), float(sigma)]
        if isinstance(sigma, (list, tuple)) and len(sigma) == 1:
            sigma = [sigma[0], sigma[0]]
        if len(sigma) != 2:
            raise ValueError(f"If sigma is a sequence, its length should be 2. Got {len(sigma)}")
        for s in sigma:
            if s <= 0.0:
                raise ValueError(f"sigma should have positive values. Got {sigma}")
        self.kernel_size, self.sigma = kernel_size, sigma
        self.dtype, self.device = dtype, device

        kernel2d = self._get_gaussian_kernel2d() # Pre-loaded kernel
        # self.variance_normalizer = torch.sum(torch.square(self.kernel2d))
        self.variance_normalizer = kernel2d.square().sum().sqrt()
        self.kernel2d = kernel2d / self.variance_normalizer
            
    def _get_gaussian_kernel1d(self, kernel_size, sigma) -> torch.Tensor:
        ksize_half = (kernel_size - 1) * 0.5
        x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel1d = pdf / pdf.sum()
        return kernel1d
            
    def _get_gaussian_kernel2d(self) -> torch.Tensor:
        kernel1d_x = self._get_gaussian_kernel1d(self.kernel_size[0], self.sigma[0]).to('cuda', dtype=self.dtype)
        kernel1d_y = self._get_gaussian_kernel1d(self.kernel_size[1], self.sigma[1]).to('cuda', dtype=self.dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :]).to(self.device)
        return kernel2d

    def __call__(self, x):
        kernel = self.kernel2d
        kernel = kernel.expand(x.shape[-3], 1, kernel.shape[0], kernel.shape[1])
        padding = [self.kernel_size[0] // 2, self.kernel_size[0] // 2, self.kernel_size[1] // 2, self.kernel_size[1] // 2]
        x = torch.nn.functional.pad(x, padding, mode="reflect")
        x = torch.conv2d(x, kernel)     # groups=x.shape[-3]
        return x
    
    def update_kernel_type(self, dtype: torch.dtype, device: torch.device=None):
        device = self.kernel2d.device if device is None else device
        self.kernel2d = self.kernel2d.to(device, dtype=dtype)
