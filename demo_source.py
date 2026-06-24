import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from demo_cmld import (
    DemoHumanML3DDataModule,
    convert_mp4_to_gif,
    load_checkpoint_state_dict,
    t2m_kinematic_chain,
)
from mld.config import parse_args
from mld.data.humanml.utils.plot_script import plot_3d_motion
from mld.models.get_model import get_model
from mld.utils.demo_utils import load_example_input
from mld.utils.logger import create_logger
from mld.utils.temos_utils import remove_padding


def extract_prefixed_state_dict(state_dict, prefix):
    extracted = OrderedDict()
    prefix_with_dot = prefix + "."
    for key, value in state_dict.items():
        if key.startswith(prefix_with_dot):
            extracted[key.replace(prefix_with_dot, "", 1)] = value
    if not extracted:
        raise RuntimeError("No checkpoint tensors found for prefix '{}'.".format(prefix))
    return extracted


def load_source_checkpoint(model, checkpoint_path, logger):
    if not checkpoint_path:
        raise RuntimeError("Source demo requires TRAIN.PRETRAINED_MLD to be set.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "Source demo checkpoint does not exist: {}".format(checkpoint_path))

    logger.info("Loading source MLD checkpoint from {}".format(checkpoint_path))
    state_dict = load_checkpoint_state_dict(checkpoint_path)

    vae_dict = extract_prefixed_state_dict(state_dict, "vae")
    model.vae.load_state_dict(vae_dict, strict=True)

    if not hasattr(model.denoiser, "mld_denoiser"):
        raise RuntimeError("Expected ControlMldDenoiser with an mld_denoiser module.")
    denoiser_dict = extract_prefixed_state_dict(state_dict, "denoiser")
    model.denoiser.mld_denoiser.load_state_dict(denoiser_dict, strict=True)


def prompt_slug(text, max_length=48):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return slug[:max_length] or "prompt"


def generate_source_motion(model, batch):
    texts = batch["text"]
    lengths = batch["length"]

    uncond_tokens = [""] * len(texts)
    uncond_tokens.extend(texts)
    text_emb = model.text_encoder(uncond_tokens)

    z = model._diffusion_reverse(
        text_emb,
        lengths,
        reference_motion=None,
        mode="v0",
    )
    feats = model.vae.decode(z, lengths)
    joints = model.feats2joints(feats.detach().cpu())
    return remove_padding(joints, lengths), feats.detach().cpu()


def main():
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.NAME = cfg.NAME + "_SOURCE"

    # Source demo is intentionally text-only. Keep these forced after CLI parsing
    # so accidental demo_source.sh overrides cannot re-enable style transfer.
    cfg.model.is_control = False
    cfg.model.is_guidance = False
    cfg.model.guidance_mode = "v0"
    cfg.model.cmld = False
    cfg.TRAIN.ABLATION.CMLD = False

    logger = create_logger(cfg, phase="demo_source")

    text, length = load_example_input(cfg.DEMO.EXAMPLE)
    if not text:
        raise ValueError("Source demo requires at least one prompt in {}.".format(
            cfg.DEMO.EXAMPLE))

    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = DemoHumanML3DDataModule(cfg)
    cfg.DATASET.NFEATS = dataset.nfeats
    cfg.DATASET.NJOINTS = dataset.njoints

    model = get_model(cfg, dataset)
    load_source_checkpoint(model, cfg.TRAIN.PRETRAINED_MLD, logger)
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = {"length": length, "text": text}
    logger.info("Begin source Text2Motion demo")

    total_start = time.time()
    with torch.no_grad():
        for rep in range(cfg.DEMO.REPLICATION):
            infer_start = time.time()
            joints, feats = generate_source_motion(model, batch)
            infer_time = time.time() - infer_start

            for i in range(len(joints)):
                base_name = "{}_{}_rep{}".format(
                    i, length[i], rep)
                slug = prompt_slug(batch["text"][i])
                npypath = output_dir / "{}_{}.npy".format(base_name, slug)
                featpath = output_dir / "{}_{}_feat.npy".format(base_name, slug)

                np.save(str(npypath), joints[i].detach().cpu().numpy())
                np.save(str(featpath), feats[i, :length[i]].detach().cpu().numpy())
                logger.info("Source motion saved:\n{}".format(npypath))

                fig_path = Path(str(npypath).replace(".npy", ".mp4"))
                gif_path = Path(str(npypath).replace(".npy", ".gif"))
                plot_3d_motion(
                    fig_path,
                    t2m_kinematic_chain,
                    joints[i].detach().cpu().numpy(),
                    title=batch["text"][i],
                    dataset="humanml",
                    fps=20,
                )
                convert_mp4_to_gif(str(fig_path), str(gif_path))

            num_all_frame = sum(batch["length"])
            print("Source Infer time - rep {}: {:.2f}s".format(rep, infer_time))
            print("Source Infer FPS - rep {}: {:.2f}".format(
                rep, num_all_frame / infer_time))

    print("Total time spent: {:.2f} seconds.".format(time.time() - total_start))


if __name__ == "__main__":
    main()
