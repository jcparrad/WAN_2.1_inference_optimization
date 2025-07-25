# generate_vace_multi.py

import os
import torch
import torch.distributed as dist
import logging
import sys
from datetime import datetime

from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import cache_video
import wan

# === Configuración de parámetros fijos ===
task = "vace-14B"
size = "832*480"
ckpt_dir = "/workspace/Wan2.1-VACE-14B"
ulysses_size = 4
ring_size = 1
t5_fsdp = True
dit_fsdp = True
frame_num = 81
sample_shift = 16
sample_solver = "unipc"
sampling_steps = 50
guide_scale = 5.0
base_seed = 42

# === Lista de tareas ===
tasks = [
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_1.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_2.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_3.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_4.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_5.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_6.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_7.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_8.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_9.mp4"
    },
    {
        "prompt": "A wide modern kitchen environment featuring a black Oster blender filled with fruits and ice on a white marble counter. Extend the current countertop seamlessly to the left and right, adding clean kitchen utensils, fresh fruit bowls, and a cutting board. Behind the blender, a tiled backsplash and white cabinets continue across the scene. Include soft natural lighting and subtle reflections on the countertop for a realistic look. Maintain the color tones and material style of the original image.",
        "src_ref_images_path": "/workspace/inference_optimization/licuadora1.png",
        "save_file": "/workspace/inference_optimization/output/output_licuadora_832_480_v2_10.mp4"
    },
]

# === Configuración de entorno distribuido ===
rank = int(os.getenv("RANK", 0))
world_size = int(os.getenv("WORLD_SIZE", 1))
local_rank = int(os.getenv("LOCAL_RANK", 0))
device = local_rank

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

# === Inicializar grupos de paralelismo para Wan2.1 ===
if ulysses_size > 1 or ring_size > 1:
    from xfuser.core.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=ring_size,
        ulysses_degree=ulysses_size,
    )

# === Logging (solo rank 0) ===
if rank == 0:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
else:
    logging.basicConfig(level=logging.ERROR)

# === Cargar configuración y modelo WanVace ===
cfg = WAN_CONFIGS[task]
wan_vace = wan.WanVace(
    config=cfg,
    checkpoint_dir=ckpt_dir,
    device_id=device,
    rank=rank,
    t5_fsdp=t5_fsdp,
    dit_fsdp=dit_fsdp,
    use_usp=(ulysses_size > 1 or ring_size > 1),
    t5_cpu=False,
)

# === Procesar tareas ===
for i, task_item in enumerate(tasks):
    if rank == 0:
        logging.info(f"[{i+1}/{len(tasks)}] Generando video para prompt: {task_item['prompt']}")

    src_video, src_mask, src_ref_images = wan_vace.prepare_source(
        [None],
        [None],
        [[task_item["src_ref_images_path"]]],
        frame_num,
        SIZE_CONFIGS[size],
        device
    )

    video = wan_vace.generate(
        task_item["prompt"],
        src_video,
        src_mask,
        src_ref_images,
        size=SIZE_CONFIGS[size],
        frame_num=frame_num,
        shift=sample_shift,
        sample_solver=sample_solver,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=base_seed,
        offload_model=False
    )

    if rank == 0:
        logging.info(f"Guardando video en {task_item['save_file']}")
        cache_video(
            tensor=video[None],
            save_file=task_item["save_file"],
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )

if rank == 0:
    logging.info("✅ Todas las tareas fueron completadas.")
