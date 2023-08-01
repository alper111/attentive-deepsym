import sys

import torch

from models import DeepSym, AttentiveDeepSym, MultiDeepSym, load_ckpt
from dataset import StateActionEffectDataset

model_types = {
    "attentive": AttentiveDeepSym,
    "multideepsym": MultiDeepSym,
    "deepsym": DeepSym
}

if __name__ == "__main__":
    model_type = model_types[sys.argv[1]]
    dataset_name = "blocks4_l"
    avg_errors = []
    for arg in sys.argv[2:]:
        model, _ = load_ckpt(arg, model_type=model_type, tag="best")
        model.freeze()
        dataset = StateActionEffectDataset(dataset_name, split="test", obj_relative=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        errors = []
        for sample in loader:
            error = model.test_step(sample, None)
            errors.append(error)
        errors = torch.cat(errors, dim=0)
        avg_error = errors.mean(dim=[0, 1]).sum()
        avg_errors.append(avg_error)
        print(f"{arg}: {avg_error*100}")
    avg_errors = torch.stack(avg_errors, dim=0)
    print(f"{avg_errors.mean(dim=0)*100:.2f} +- {avg_errors.std(dim=0)*100:.2f}")
