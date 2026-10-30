"""Encode a semantics config's CLIP prompts into class prototypes. Run with LangSplat's environment (open_clip):

    env -u PYTHONPATH ../.venv-langsplat/bin/python scripts/clip-prototypes.py configs/semantics-v3.json OUT.npz

Uses the same OpenCLIP model as LangSplat's preprocess.py, on CPU, from the local Hugging Face cache.
"""

import json
import sys

import numpy as np
import open_clip
import torch

MODEL, PRETRAINED = 'ViT-B-16', 'laion2b_s34b_b88k'


def main(config_path, output):
    prompts = json.loads(open(config_path).read())['prompts']
    model, _, _ = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL)
    vectors = []
    with torch.no_grad():
        for phrases in prompts.values():
            embedded = torch.nn.functional.normalize(model.encode_text(tokenizer(phrases)).float(), dim=-1)
            vectors.append(torch.nn.functional.normalize(embedded.mean(dim=0), dim=0).numpy())
    np.savez(output, names=list(prompts), vectors=np.stack(vectors), prompts=json.dumps(prompts),
             model=f'{MODEL}/{PRETRAINED}')
    print(f'{len(prompts)} class prototypes written to {output}')


if __name__ == '__main__':
    main(*sys.argv[1:])
