import torch
import torch as torch
import torch.nn.functional as F
import torch.nn as nn
from src.model.feat_extractor import FeatExtractor
from src.model.predictor import Predictor
import wandb


if __name__ == "__main__":
    wandb.init(project="oc-data-collection", entity='atari-obj-pred', name="debug")
    batch_size = 1
    num_obj = 4
    images = (256* torch.rand((batch_size, 4, 3, 128, 128))).int().cuda()
    images = torch.arange(0, batch_size, step=1, device='cuda').repeat_interleave(4*3*128*128).reshape((batch_size, 4, 3, 128, 128))
    objects = (num_obj*torch.rand((batch_size, 128, 128))).int().cuda()
    # objects = torch.arange(0, num_obj, step=1).repeat_interleave(128*128).reshape((num_obj, 128, 128)).int().cuda()
    wandb.log({'batch_size':batch_size})
    feat_extract = FeatExtractor(num_objects=num_obj).cuda()
    #wandb.watch(feat_extract, log="all", log_freq=1, idx=1)
    t_step = 1
    predictor = Predictor(num_layers=1, time=t_step).cuda()
    #wandb.watch(predictor, log="all", log_freq=1, idx=2)
    target = torch.rand((batch_size, t_step, num_obj, 2)).cuda()
    # target = torch.ones((batch_size, t_step, 32, 2)).cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(list(feat_extract.parameters()) + list(predictor.parameters()), lr=1e-3)

    ooutput = torch.rand((batch_size, num_obj, 128)).cuda()
    for i in range(1000):
        output = feat_extract(images, objects)
        output = predictor(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(loss.item(), output.mean().item(), output.std().item())
        wandb.log({"loss":loss})
        optimizer.zero_grad()

    print(target)
    print(output)


