import torch
import utils
import geex, competitors
from tqdm import tqdm
from torchvision.transforms import GaussianBlur

# A unified interface handling the output of various explainers
def unified_expl(compoenents: list):
    """
    Parameters:
        expl_type:  a string specifying the explainer to be tested
        components: a list of components that are necessary to deriving explanations,
                    containing [data, label, model, explainer]
    """
    data, lbl, m, expl = compoenents 
    if isinstance(expl, geex.GE):
        rel = expl.explain(m, data)
        rel = torch.mean(rel, dim=0, keepdim=True)
    elif isinstance(expl, competitors.RISE):
        rel = expl(data.cuda())[lbl] if hasattr(expl, 'gpu_batch') else expl.explain(m, data)[lbl]
    elif isinstance(expl, competitors.GradBased):
        rel = expl.explain(m, data)
        rel = torch.mean(rel, dim=0)
    else:
        assert 1 == 0, f'Unknown explainer type: [{expl.__class__}]'
    return rel

# Sorting features (pixels) in descending order
def get_pix_ranking(rel):
    rel = rel.flatten()
    pairs = zip(rel, list(range(len(rel))))
    return sorted(pairs, key=lambda x:x[0], reverse=True)

def expl_guided_edition(m, img_org, rel, lbl, num=100, v_default=0, stride=1, batch_size=64, pbar=None):
    """
    Parameters:
        m: model
        img_org: torch.Tensor in shape [1, C, H, W]
        rel: attribution matrix in shape [H, W]
    """
    img = img_org.clone()
    pix_ranking = get_pix_ranking(rel)

    pred_manipul = [m(img.cuda()).detach().cpu()[0][lbl]]
    ptr = 0
    flag_subtask = pbar is not None
    pbar = pbar if pbar is not None else tqdm(total=num)

    while ptr < num:
        buff = []
        while len(buff) < batch_size:
            start, end = ptr, min(ptr + stride, num)
            img = mask_out_pix(img, pix_ranking[start: end], v_default)
            buff.append(img.clone())
            ptr += stride
            if ptr >= num: break
        preds = list(m(torch.cat(buff).to('cuda')).cpu().detach()[:, lbl])
        pred_manipul += preds

        if flag_subtask: 
            pbar.set_postfix_str('Current edition: [{}/{}]'.format(ptr, num))
        else:
            pbar.update(batch_size)
    return pred_manipul, img

def evaluate_via_deletion(expl, m, test_loader, input_shape, del_ratio=1.0, v_default=0.1307, 
                          num_instances=None, stride=10, batch_size=64):
    """
    Parameters:
        expl: explainer instance
        m: the to-be-explained model
        input_shape:    [tuple] required for the preparation of the default value matrix
        del_ratio:      [float] the ratio of features to be removed, 1.0 indicates total removal
        v_default: the value for replacement, supported values:
            1. [int/float] identical default value for all features with single constant
            2. [torch.Tensor] of shape [C, H, W], Gaussian noise 
            3. [str] 'blur' specifying a blurred version of explicand as absence values
        stride: [int] the period (in steps of deletion) of querying the model for confidence drops 
    """
    v_default = prepare_default(v_default, input_shape)
    num_del = int(input_shape[-1] * input_shape[-2] * del_ratio)
    test_handler = iter(test_loader) 
    records, count = [], 0 
    num_instances = num_instances if num_instances else len(test_loader)*test_loader.batch_size
    print(f'Num of instances for evaluation: {num_instances}')

    blur_m = GaussianBlur((31, 31), 5.0)
    with tqdm(total=num_instances, desc='#Instances') as pbar:
        for batch in test_handler:
            imgs, _ = batch
            for img in imgs:
                img = img.unsqueeze(dim=0)
                v = blur_m(img) if v_default =='blur' else v_default
                v = v[0] if len(v.shape) == 4 else v
                logits = m(img.cuda()).detach().cpu()
                num_classes = len(logits[0])

                # Get the initally predicted class, whose confidence drop is supervised
                _, lbl = torch.max(logits, dim=1)
                components = [img, lbl, m] + [expl] 
                rel = unified_expl(components)

                # Start of the explanation-guided deletion process for the current explicand
                trend, _ = expl_guided_edition(m, img, rel, lbl, v_default=v, 
                                            num=num_del, stride=stride, pbar=pbar,
                                            batch_size=batch_size)
                records.append(trend)
                count += 1
                pbar.update(1)
                if count >= num_instances:
                    return records
    return records

def prepare_default(v, shape):
    if isinstance(v, str):
        v = v.lower()
    elif isinstance(v, int) or isinstance(v, float):
        v = torch.tile(torch.tensor(v), (1,)+shape[-2:])
    return v

def mask_out_pix(img, idxs, v_default):
    for _, idx in idxs:
        rid, cid = utils.idx2pos(idx, img.shape[-1]) 
        v = v_default if isinstance(v_default, (int, float)) else v_default[:, rid, cid]
        img[0,:,rid,cid] = v
    return img

def compute_aopc(trend):
    return 1. - torch.mean(trend) / trend[0]
