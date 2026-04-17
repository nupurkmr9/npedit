from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
import gc
import os
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW

# Local imports
from data import DataStreamer
from models.latent_fm import FMDataContext, FrozenOps, InferenceOps, LatentFM, TrainableOpsDMD, TrainableOpsDMDAux
from models.latent_fm_factory import create_latent_fm, create_critic
from utils.ckpt import FSDPCheckpointer, load_checkpoint_for_inference
from utils.clip_grad import clip_grad
from torch.distributed._tensor import DTensor
from utils.config import ConfigurableModule
from utils.ema import copy_params, update_ema
from utils.fsdp import fwd_only_mode
from utils.log import TrackingLogger, WandbLogger, get_logger, get_pbar, human_readable_number
from utils.lr import LinearWarmupCosineDecayScheduler
from utils.optim import create_parameter_groups
from utils.prof import Profiler
from data import MiniBatchDataLoader

from . import BaseTrainerParams, setup_distributed, setup_experiment_dirs

logger = get_logger(__name__)


@dataclass
class DITTrainerParams(BaseTrainerParams):
    """Parameters for DiTTrainer."""

    # Text dropout probability
    txt_drop_prob: float = 0.1
    aux_local_steps_per_global_step: int = 10
    lora_rank: int = 0
    lora_rank_aux: int = 0

    init_ckpt_student: str | None = None
    init_ckpt_load_plan_student: str | None = None
    init_ckpt_aux: str | None = None
    init_ckpt_load_plan_aux: str | None = None
    init_ckpt_teacher: str | None = None
    init_ckpt_load_plan_teacher: str | None = None

    start_multistep_sampling_step: int = 4000

    # loss hyperparameters
    critic_loss_weight: float = 0.01
    dmd_loss_weight: float = 0.5

    # Decoupled DMD (arXiv 2511.22677)
    decoupled_dmd: bool = False
    ca_loss_weight: float = 1.0
    dm_loss_weight: float = 1.0

    data_seed: int = 42


class DMDTrainer(ConfigurableModule[DITTrainerParams]):
    """Distributed Flow Matching trainer."""

    # ---- Components used for training ----
    params: DITTrainerParams | None = None
    config: dict[str, Any] | None = None

    latent_fm: LatentFM | None = None
    aux_fm: LatentFM | None = None
    teacher_fm: LatentFM | None = None
    frozen_ops: FrozenOps | None = None
    trainable_ops: TrainableOpsDMD | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    checkpointer: FSDPCheckpointer | None = None

    device: torch.device | None = None
    local_rank: int | None = None
    global_rank: int | None = None
    world_size: int | None = None

    exp_dir: str | None = None
    run_dir: str | None = None
    ckpt_dir: str | None = None

    step: int | None = None
    start_step: int | None = None

    def __init__(self, params: DITTrainerParams):
        self.params = params
        logger.info(f"Initialized DiTTrainer with params: {params}")

    @classmethod
    def get_default_params(cls) -> DITTrainerParams:
        """Return the default parameters for DiTTrainer."""
        return DITTrainerParams()

    @property
    def training_state_student(self) -> dict[str, Any]:
        return {
            "model": self.latent_fm.denoiser,
            "ema": self.latent_fm.ema_denoiser,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }
    
    @property
    def training_state_aux(self) -> dict[str, Any]:
        return {
            "model": self.aux_fm.denoiser,
            "ema": self.aux_fm.ema_denoiser,
            "optimizer": self.aux_optimizer,
            "scheduler": self.aux_scheduler,
        }

    @property
    def training_state_teacher(self) -> dict[str, Any]:
        return {
            "model": self.teacher_fm.denoiser,
            "ema": self.teacher_fm.ema_denoiser,
        }

    @property
    def training_meta_student(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "world_size": self.world_size,
            "run_dir": self.run_dir,
            "ckpt_dir": f"{self.ckpt_dir}/student",
        }
    
    @property
    def training_meta_aux(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "world_size": self.world_size,
            "run_dir": self.run_dir,
            "ckpt_dir": f"{self.ckpt_dir}/aux",
        }


    @property
    def denoiser(self) -> nn.Module:
        return self.latent_fm.denoiser
    
    @property
    def aux_denoiser(self) -> nn.Module:
        return self.aux_fm.denoiser

    @property
    def ema_denoiser(self) -> nn.Module:
        return self.latent_fm.ema_denoiser

    def train(self, config: dict[str, Any]) -> None:
        """Train the model."""
        self.config = config

        # Initialize training
        self.train_init()

        # Run training loop
        self.train_loop()

        # Clean up distributed training
        dist.destroy_process_group()

    def train_init(self) -> None:
        # Use rank-invariant seed up util the creation of dataloader
        torch.manual_seed(0)

        # Initialize distributed training
        self.device, self.local_rank, self.global_rank, self.world_size = setup_distributed()

        # Setup experiment directories
        self.exp_dir = self.params.exp_dir
        self.run_dir, self.ckpt_dir = setup_experiment_dirs(self.exp_dir, self.config)

        # Create LatentFM with all components
        logger.info("Creating LatentFM with all components...")
        self.latent_fm = create_latent_fm(self.config["student"], self.device, create_ema=False, mode='student', lora_rank=self.params.lora_rank)
        self.aux_fm = create_latent_fm(self.config["aux"], self.device, create_ema=False, mode='aux', lora_rank=self.params.lora_rank_aux)
        self.teacher_fm = create_latent_fm(self.config["teacher"], self.device, create_ema=False, mode='teacher')

        if "critic" in self.config:
            self.critic = create_critic(self.config["critic"], self.device)
        else:
            logger.info("No critic model configured")
            self.critic = None

        # Ensure the required trainable modules are present (type safety for mypy)
        assert self.latent_fm.denoiser is not None, "Denoiser must be provided by create_latent_fm"

        # Create FrozenOps and TrainableOps
        logger.info("Setting up FrozenOps and TrainableOps...")
        self.frozen_ops = FrozenOps(lfm=self.latent_fm)

        self.trainable_ops = TrainableOpsDMD(lfm=self.latent_fm, aux_lfm=self.aux_fm, teacher_lfm=self.teacher_fm, critic=self.critic, global_batch_size=None, critic_loss_weight=self.params.critic_loss_weight, dmd_loss_weight=self.params.dmd_loss_weight, start_multistep_sampling_step=self.params.start_multistep_sampling_step, image_log_freq=self.params.image_log_freq)

        self.aux_trainable_ops = TrainableOpsDMDAux(lfm=self.latent_fm, aux_lfm=self.aux_fm, global_batch_size=None, image_log_freq=self.params.image_log_freq, start_multistep_sampling_step=self.params.start_multistep_sampling_step)



        # Initialize checkpointer
        logger.info("Setting up checkpointer...")
        self.checkpointer = FSDPCheckpointer(os.path.join(self.ckpt_dir, "student"))
        self.checkpointer_aux = FSDPCheckpointer(os.path.join(self.ckpt_dir, "aux"))
        self.checkpointer_teacher = FSDPCheckpointer(os.path.join(self.ckpt_dir, "teacher"))

        # Create optimizer (only for trainable denoiser parameters)
        logger.info("Setting up optimizer...")
        self.optimizer = AdamW(
            create_parameter_groups(self.denoiser, self.params.weight_decay),
            lr=self.params.max_lr,
            betas=self.params.adam_betas,
            fused=True,
        )

        self.aux_optimizer = AdamW(
            create_parameter_groups(self.aux_denoiser, self.params.weight_decay_aux),
            lr=self.params.max_lr_aux,
            betas=self.params.adam_betas_aux,
            fused=True,
        )

        # Create learning rate scheduler
        logger.info("Setting up learning rate scheduler...")
        self.scheduler = LinearWarmupCosineDecayScheduler(
            optimizer=self.optimizer,
            warmup_steps=self.params.warmup_steps,
            total_steps=self.params.max_steps,
            max_lr=self.params.max_lr,
            min_lr=self.params.min_lr,
        )

        self.aux_scheduler = LinearWarmupCosineDecayScheduler(
            optimizer=self.aux_optimizer,
            warmup_steps=self.params.warmup_steps,
            total_steps=self.params.max_steps,
            max_lr=self.params.max_lr_aux,
            min_lr=self.params.min_lr_aux,
        )

        # Switch to rank-dependent seed
        # Note that checkpointer may restore RNG state from checkpoint if it exists
        torch.manual_seed(42 + self.global_rank)

        # Resume from latest checkpoint if available
        logger.info("Trying to resume from latest checkpoint...")
        self.start_step = 0
        self.start_step = self.checkpointer.resume_latest(
            **{"model": self.latent_fm.denoiser, "ema": self.latent_fm.ema_denoiser, "optimizer": self.optimizer, "scheduler": self.scheduler},
            init_ckpt=self.params.init_ckpt_student,
            init_ckpt_load_plan=self.params.init_ckpt_load_plan_student,
        )
        _ = self.checkpointer_aux.resume_latest(
            **self.training_state_aux,
            init_ckpt=self.params.init_ckpt_aux,
            init_ckpt_load_plan=self.params.init_ckpt_load_plan_aux,
        )
        _ = self.checkpointer_teacher.resume_latest(
            **self.training_state_teacher,
            init_ckpt=self.params.init_ckpt_teacher,
            init_ckpt_load_plan=self.params.init_ckpt_load_plan_teacher,
        )
        if self.start_step == 0 and self.ema_denoiser is not None:
            # Initialize EMA with main model parameters
            logger.info("Initializing EMA with main model parameters...")
            copy_params(self.denoiser, self.ema_denoiser, model_to_ema=True)
        self.step = self.start_step


        # Initialize wandb logger
        self.wandb_logger = WandbLogger(
            project=self.params.wandb_project,
            name=self.params.wandb_name,
            config=self.training_meta_student,
            entity=self.params.wandb_entity,
            host=self.params.wandb_host,
            save_dir=self.run_dir,
            mode=self.params.wandb_mode,
        )

        # Create infinite dataloader
        logger.info("Setting up data streamer...")
        self.data_module = DataStreamer(self.config["data"], data_seed=self.params.data_seed + self.start_step)
        self.train_dataloader = self.data_module.train_dataloader()
        self.val_dataloader = self.data_module.val_dataloader()

        # Inference loop
        if self.params.inference_at_start:
            logger.info(f"Running inference at start step {self.start_step}...")
            self.in_train_inference()
            if self.params.inference_then_exit:
                logger.info("Finished inference and not continuing training, exiting...")
                dist.destroy_process_group()
                return

    def _warmup_fsdp_models(self) -> None:
        """Warmup FSDP models to initialize communication patterns safely."""
        with torch.no_grad():
            # Create dummy data for warmup
            dummy_prompts = ["warmup"] * 2
            dummy_images = torch.randn(2, 3, 1, 64, 64, device=self.device)
            
            dummy_context = FMDataContext(prompts=dummy_prompts, images=dummy_images)
            pixel_values = self.critic.image_prep(dummy_context.reference_images)
            pixel_values = (torch.cat([pixel_values, pixel_values], dim=0)).contiguous()                     

            

            # Warmup frozen ops (T5, CLIP) 
            try:
                self.frozen_ops(dummy_context, txt_drop_prob=0.0)
                # Warmup critic with cached ViT path
                vit_embeds = self.critic.extract_vit_features(pixel_values.contiguous())
                flags = torch.ones(vit_embeds.shape[0], device=vit_embeds.device, dtype=torch.long)
                self.critic({'input_ids': dummy_context.input_ids_identity,
                'labels': dummy_context.labels_identity,
                'image_flags': flags,
                }, temperature=1.0, global_step=0,
                vit_embeds=vit_embeds)
                logger.info("FSDP frozen ops warmup successful")
            except Exception as e:
                logger.warning(f"FSDP warmup failed: {e}")
                
            dist.barrier()

    def train_loop(self) -> None:
        """Run the training loop."""
        # Main training loop
        logger.info("Running memory cleanup before entering training loop...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        gc.disable()


        self.tracking_logger = TrackingLogger()
        self._student_param_stats: dict[str, float] = {}
        self._aux_param_stats: dict[str, float] = {}
        self.mini_batch_dataloader = MiniBatchDataLoader(
                self.data_module, self.train_dataloader, embed_fn=self.frozen_ops, start_step=self.start_step, total_batch_size=self.params.total_batch_size, tracking_logger=self.tracking_logger, txt_drop_prob=self.params.txt_drop_prob
            )
        # Initialize progress bar for rank 0
        pbar = get_pbar(self.params.max_steps, self.start_step)

        if dist.is_initialized():
            logger.info("Synchronizing all ranks before training...")
            dist.barrier()

        logger.info(f"Starting training from step {self.start_step} for {self.params.max_steps} total steps")
        trainable_params_student = [p for _, p in self.denoiser.named_parameters() if p.requires_grad]
        trainable_params_aux = [p for _, p in self.aux_denoiser.named_parameters() if p.requires_grad]
        
        while self.step < self.params.max_steps:
            
            for param in trainable_params_aux:
                param.requires_grad = True
            for param in trainable_params_student:
                param.requires_grad = False
            self.aux_denoiser.train()
            self.denoiser.eval()
            self.aux_optimizer.zero_grad(set_to_none=True)
            for i in range(self.params.aux_local_steps_per_global_step):
                self.train_one_step_aux(local_step=i, log_images = i == self.params.aux_local_steps_per_global_step-1)

            for param in trainable_params_student:
                param.requires_grad = True
            for param in trainable_params_aux:
                param.requires_grad = False
            self.denoiser.train()
            self.aux_denoiser.eval()
            if self.ema_denoiser is not None:
                self.ema_denoiser.eval()  # EMA model should be in eval mode
            self.optimizer.zero_grad(set_to_none=True)
            self.train_one_step()

            # Periodic garbage collection and synchronization across GPUs to prevent stragglers
            if self.step % self.params.gc_freq == 0:
                logger.info(f"Step {self.step:6d} | Running garbage collection...")
                # gc.collect()
                torch.cuda.empty_cache()
            
            # Logging
            if self.step % self.params.log_freq == 0:
                self.log_metrics()

            # Update progress bar for rank 0
            self.step += 1
            if pbar is not None:
                pbar.update(1)

        # Save final checkpoint
        if self.step % self.params.ckpt_freq != 0 and self.step > 0:  # Avoid duplicate checkpoint saving
            self.save_checkpoint(mode='student')
            self.save_checkpoint(mode='aux')
        self.checkpointer.finish()
        self.checkpointer_aux.finish()
        # Close progress bar
        if pbar is not None:
            pbar.close()

    
    def train_one_step_aux(self, local_step: int, log_images: bool) -> None:

        prof = None
        dump_trace = False # self.step == self.start_step + 10
        if dump_trace:
            prof = Profiler()
            prof.start()

        train_step_tic = time.time()

        self.run_fwdbwd_aux(
            mini_batch_dataloader=self.mini_batch_dataloader,
            tracking_logger=self.tracking_logger,
            print_data_summary=False,
            log_images=log_images,
        )

        # Gradient clipping
        grad_norm = clip_grad(self.aux_denoiser, self.params.gradient_clip_norm)

        # Gradient norm spike detection
        has_bad_grad = (not torch.isfinite(grad_norm)) or (
            self.step >= self.params.grad_norm_spike_detection_start_step
            and grad_norm > self.params.grad_norm_spike_threshold
        )
        if has_bad_grad and self.global_rank == 0:
            logger.warning(
                f"Step {self.step}: detected nan/inf/spiky gradient (norm={grad_norm}); skipping optimizer step."
            )

        # Optimizer step (only if gradient is healthy)
        lr = self.aux_scheduler.get_last_lr()[0]
        if not has_bad_grad:
            self.aux_optimizer.step()

        # Collect per-parameter stats before gradients are zeroed (only at log steps)
        if self.step % 5 * self.params.log_freq == 0 and self.global_rank == 0:
            self._aux_param_stats = self._collect_param_stats(self.aux_denoiser, "aux")

        # Zero out gradients
        self.aux_optimizer.zero_grad(set_to_none=True)

        # Always step the scheduler regardless of whether optimizer step was skipped
        self.aux_scheduler.step()

        # Update tracking logger
        self.tracking_logger.log(
            {
                "aux/has_bad_grad": has_bad_grad,
                "aux/lr": lr,
                "aux/grad_norm": grad_norm,
                "aux/step_duration": time.time() - train_step_tic,
            }
        )

        if dump_trace:
            trace_path = (
                os.path.join(self.run_dir, f"traces/aux/rank_{self.global_rank}.json") if self.global_rank == 0 else None
            )
            prof.stop(trace_path)

        # Print global batch size
        if self.step == self.start_step + 1:
            logger.info(f"Global batch size: {self.aux_trainable_ops.global_batch_size}")

        # Save checkpoint
        if self.step % self.params.ckpt_freq == 0 and self.step > 0:
            self.save_checkpoint(mode='aux')

    def run_fwdbwd_aux(
        self,
        mini_batch_dataloader: MiniBatchDataLoader,
        tracking_logger: TrackingLogger,
        skip_backward: bool = False,
        print_data_summary: bool = False,
        log_images: bool = False,
    ) -> torch.Tensor:
        
        for batch in mini_batch_dataloader.step(tracking_logger=tracking_logger):
            mode = 'aux'
            fm_data_context = batch.fm_data_context
            
            # Print data summary if requested
            if print_data_summary:
                fm_data_context.summarize()

            # Step 2: Process through TrainableOps (Denoiser + Loss computation)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # <-- ADD THIS     
                with tracking_logger.log_time(f"time/{mode}/trainable_ops_fwd"):
                    # with fwd_only_mode(self.denoiser):
                    #     fm_data_context = self.get_fake_samples_from_student_model(fm_data_context, global_step=self.step)
                            
                    fm_data_context = self.aux_trainable_ops(fm_data_context, global_step=self.step, log_images=log_images)

            if not skip_backward:
                with tracking_logger.log_time(f"time/{mode}/trainable_ops_bwd"):
                    scaled_loss = fm_data_context.loss / mini_batch_dataloader.num_accum_steps
                    scaled_loss.backward()

            tracking_logger.log({f"{mode}/loss_vec": fm_data_context.loss_vec, f"{mode}/num_tokens": fm_data_context.num_tokens})


    def train_one_step(self) -> None:

        prof = None
        dump_trace = False # self.step == self.start_step + 10
        if dump_trace:
            prof = Profiler()
            prof.start()

        train_step_tic = time.time()

        self.run_fwdbwd(
            mini_batch_dataloader=self.mini_batch_dataloader,
            tracking_logger=self.tracking_logger,
            print_data_summary=False,
        )

        # Gradient clipping
        grad_norm = clip_grad(self.denoiser, self.params.gradient_clip_norm)

        # Gradient norm spike detection
        has_bad_grad = (not torch.isfinite(grad_norm)) or (
            self.step >= self.params.grad_norm_spike_detection_start_step
            and grad_norm > self.params.grad_norm_spike_threshold
        )
        if has_bad_grad and self.global_rank == 0:
            logger.warning(
                f"Step {self.step}: detected nan/inf/spiky gradient (norm={grad_norm}); skipping optimizer step."
            )

        # Optimizer step (only if gradient is healthy)
        lr = self.scheduler.get_last_lr()[0]
        if not has_bad_grad:
            self.optimizer.step()
            if self.ema_denoiser is not None:
                update_ema(self.denoiser, self.ema_denoiser, self.params.ema_decay)

        # Collect per-parameter stats before gradients are zeroed (only at log steps)
        if self.step % 5 * self.params.log_freq == 0 and self.global_rank == 0:
            self._student_param_stats = self._collect_param_stats(self.denoiser, "student")

        # Zero out gradients
        self.optimizer.zero_grad()

        # Always step the scheduler regardless of whether optimizer step was skipped
        self.scheduler.step()

        # Update tracking logger
        self.tracking_logger.log(
            {
                "student/has_bad_grad": has_bad_grad,
                "student/lr": lr,
                "student/grad_norm": grad_norm,
                "student/step_duration": time.time() - train_step_tic,
            }
        )

        if dump_trace:
            trace_path = (
                os.path.join(self.run_dir, f"traces/student/rank_{self.global_rank}.json") if self.global_rank == 0 else None
            )
            prof.stop(trace_path)

        # Print global batch size
        if self.step == self.start_step + 1:
            logger.info(f"Global batch size: {self.trainable_ops.global_batch_size}")

        # Save checkpoint
        if self.step % self.params.ckpt_freq == 0 and self.step > 0:
            self.save_checkpoint(mode='student')

        # Inference loop
        # if self.step % self.params.inference_freq == 0:
        #     self.in_train_inference()

        # Validation loop
        # if self.step % self.params.val_freq == 0 and self.step > 0:
        #     self.run_validation()
        

    def run_fwdbwd(
        self,
        mini_batch_dataloader: MiniBatchDataLoader,
        tracking_logger: TrackingLogger,
        skip_backward: bool = False,
        print_data_summary: bool = False,
    ) -> torch.Tensor:
        
        for batch in mini_batch_dataloader.step(tracking_logger=tracking_logger):
            fm_data_context = batch.fm_data_context
            mode = 'student'
            
            # Print data summary if requested
            if print_data_summary:
                fm_data_context.summarize()

            # Step 2: Process through TrainableOps (Denoiser + Loss computation)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # <-- ADD THIS  
                with tracking_logger.log_time(f"time/{mode}/trainable_ops_fwd"):
                    fm_data_context = self.trainable_ops(fm_data_context, global_step=self.step)

            if not skip_backward:
                with tracking_logger.log_time(f"time/{mode}/trainable_ops_bwd"):
                    scaled_loss = fm_data_context.loss / mini_batch_dataloader.num_accum_steps
                    scaled_loss.backward()

            tracking_logger.log({f"{mode}/loss_vec": fm_data_context.loss_vec, f"{mode}/num_tokens": fm_data_context.num_tokens})
            tracking_logger.log({f"{mode}/timesteps": fm_data_context.timesteps})
            tracking_logger.log({f"{mode}/mse_loss_vec": fm_data_context.mse_loss_vec})
            tracking_logger.log({f"{mode}/dmd_loss_vec": fm_data_context.dmd_loss_vec})
            if self.critic is not None:
                tracking_logger.log({f"{mode}/score_editing": fm_data_context.score_logits_edit_eval})
                tracking_logger.log({f"{mode}/score_identity": fm_data_context.score_logits_identity_eval})
                tracking_logger.log({f"{mode}/critic_loss_vec": fm_data_context.critic_loss_vec})

    @torch.no_grad()
    def run_validation(self) -> None:
        self.denoiser.eval()

        logger.info(f"Step {self.step:6d} | Running validation...")

        val_dataloader = self.data_module.val_dataloader()
        val_tracking_logger = TrackingLogger()

        val_mini_batch_dataloader = MiniBatchDataLoader(
                self.data_module, self.val_dataloader, embed_fn=self.frozen_ops, start_step=self.start_step, total_batch_size=self.params.total_batch_size, tracking_logger=val_tracking_logger, txt_drop_prob=0.0
            )

        with fwd_only_mode(self.denoiser), val_tracking_logger.log_time("time/validation"):
            self.run_fwdbwd(
                val_mini_batch_dataloader,
                tracking_logger=val_tracking_logger,
                skip_backward=True,
            )

        val_tracking_logger.flush()
        avg_val_loss = val_tracking_logger["loss_vec", "mean"]
        max_val_time = val_tracking_logger["time/validation", "max"]

        logger.info(f"Step {self.step:6d} | Validation loss: {avg_val_loss:.4f}")

        if self.global_rank == 0:
            self.wandb_logger.log({"validation/loss": avg_val_loss, "time/validation": max_val_time}, step=self.step)

    def save_checkpoint(self, mode:str='student') -> None:
        torch.cuda.empty_cache()
        if mode == 'student':
            self.checkpointer.save_checkpoint(
            step=self.step,
                **self.training_state_student,
                metadata=self.training_meta_student,
            )
            self.checkpointer.finish()
        elif mode == 'aux':
            self.checkpointer_aux.save_checkpoint(
                step=self.step,
                **self.training_state_aux,
                metadata=self.training_meta_aux,
            )
            self.checkpointer_aux.finish()

    @torch.no_grad()
    def _collect_param_stats(self, model: nn.Module, prefix: str) -> dict[str, float]:
        """Collect per-parameter gradient and weight statistics for wandb logging."""
        stats = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Clean up FSDP wrapper names
            clean_name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")

            # Handle DTensor (FSDP-sharded) vs regular tensors
            from torch.distributed.tensor import DTensor

            if isinstance(param, DTensor):
                w = param._local_tensor
                if w.numel() == 0:
                    continue
            else:
                w = param
            w = w.detach().float()
            stats[f"weights/{prefix}/{clean_name}/norm"] = w.norm().item()
            stats[f"weights/{prefix}/{clean_name}/mean"] = w.mean().item()
            # stats[f"weights/{prefix}/{clean_name}/std"] = w.std().item()

            if param.grad is not None:
                grad = param.grad._local_tensor if isinstance(param.grad, DTensor) else param.grad
                g = grad.detach().float()
                stats[f"grad_weights/{prefix}/{clean_name}/norm"] = g.norm().item()
                stats[f"grad_weights/{prefix}/{clean_name}/mean"] = g.mean().item()
                # stats[f"grad_weights/{prefix}/{clean_name}/std"] = g.std().item()
        return stats

    def log_metrics(self) -> None:
        # Compute average loss across all batch elements and GPUs
        self.tracking_logger.flush()
        max_data_time = self.tracking_logger[f"time/data", "max"]
        max_frozen_ops_time = self.tracking_logger[f"time/frozen_ops", "max"]
        log_metrics = {'student': {}, 'aux': {}}
            
        for mode in ['student', 'aux']:
            log_metrics[mode]["avg_loss"] = self.tracking_logger[f"{mode}/loss_vec", "mean"]
            log_metrics[mode]["avg_lr"] = self.tracking_logger[f"{mode}/lr", "mean"]
            log_metrics[mode]["mean_grad_norm"] = self.tracking_logger[f"{mode}/grad_norm", "mean"]
            log_metrics[mode]["bad_grad_count"] = self.tracking_logger[f"{mode}/has_bad_grad", "sum"]
            log_metrics[mode]["tps"] = self.tracking_logger[f"{mode}/num_tokens", "sum"] / self.tracking_logger[f"{mode}/step_duration", "sum"] * self.world_size
            log_metrics[mode]["max_trainable_ops_fwd_time"] = self.tracking_logger[f"time/{mode}/trainable_ops_fwd", "max"]
            log_metrics[mode]["max_trainable_ops_bwd_time"] = self.tracking_logger[f"time/{mode}/trainable_ops_bwd", "max"]
            # check if mode is student then log the score_editing and score_identity
            if mode == 'student':
                if self.critic is not None:
                    log_metrics[mode]["avg_score_editing"] = self.tracking_logger[f"{mode}/score_editing", "mean"]
                    log_metrics[mode]["avg_score_identity"] = self.tracking_logger[f"{mode}/score_identity", "mean"]
                    log_metrics[mode]["avg_critic_loss_vec"] = self.tracking_logger[f"{mode}/critic_loss_vec", "mean"]
                log_metrics[mode]["avg_timesteps"] = self.tracking_logger[f"{mode}/timesteps", "mean"]
                log_metrics[mode]["avg_mse_loss_vec"] = self.tracking_logger[f"{mode}/mse_loss_vec", "mean"]
                log_metrics[mode]["avg_dmd_loss_vec"] = self.tracking_logger[f"{mode}/dmd_loss_vec", "mean"]

            if dist.get_rank() == 0:
                avg_loss = log_metrics[mode]["avg_loss"]
                avg_lr = log_metrics[mode]["avg_lr"]
                mean_grad_norm = log_metrics[mode]["mean_grad_norm"]
                tps = log_metrics[mode]["tps"]
                max_trainable_ops_fwd_time = log_metrics[mode]["max_trainable_ops_fwd_time"]
                max_trainable_ops_bwd_time = log_metrics[mode]["max_trainable_ops_bwd_time"]
                logger.info(
                    f"Step {self.step:6d} | Loss: {avg_loss:.4f} | LR: {avg_lr:.2e} | GradNorm: {mean_grad_norm:.2f} | "
                    f"TPS: {human_readable_number(tps)} | Data: {max_data_time:.3f}s | Frozen: {max_frozen_ops_time:.3f}s | "
                    f"TrainableFwd: {max_trainable_ops_fwd_time:.3f}s | TrainableBwd: {max_trainable_ops_bwd_time:.3f}s"
                )
            
        # Merge per-parameter gradient/weight stats
        param_stats = {**self._student_param_stats, **self._aux_param_stats}
        self._student_param_stats = {}
        self._aux_param_stats = {}

        self.wandb_logger.log(
            {**{f'student/{key}': value for key, value in log_metrics['student'].items()}, **{f'aux/{key}': value for key, value in log_metrics['aux'].items()}, "time/data": max_data_time, "time/frozen_ops": max_frozen_ops_time, **param_stats},
            step=self.step,
        )

    def in_train_inference(self) -> None:
        assert (
            self.latent_fm is not None and self.exp_dir is not None and self.step is not None
        ), "LatentFM, exp_dir, and step must be provided in training mode"

        logger.info("Setting up InferenceOps...")
        inference_ops = InferenceOps(lfm=self.latent_fm, train_dataloader=self.train_dataloader, data_module=self.data_module)

        for use_ema in [False, True]:
            logger.info(f"Running inference with EMA={int(use_ema)}...")
            config = deepcopy(self.config)
            config["inferencer"]["inference_ops_args"]["use_ema"] = use_ema
            config["inferencer"]["inference_ops_args"]["output_dir"] = os.path.join(
                self.exp_dir, f"inference_ema-{int(use_ema)}/step_{self.step:08d}"
            )
            inference_ops(**config["inferencer"]["inference_ops_args"])

    def inference(self, config: dict[str, Any]) -> None:
        self.config = config

        # Use rank-invariant seed up util the creation of dataloader
        torch.manual_seed(0)

        # Initialize distributed training
        self.device, self.local_rank, self.global_rank, self.world_size = setup_distributed()

        # Create LatentFM with all components
        logger.info("Creating LatentFM with all components...")
        self.latent_fm = create_latent_fm(self.config["student"], self.device, create_ema=False, mode='student', lora_rank=self.params.lora_rank)

        # Ensure the required modules are present (type safety for mypy)
        assert self.latent_fm.denoiser is not None, "Denoiser must be provided by create_latent_fm"

        ckpt_dir = config["inferencer"]["ckpt_dir"]
        use_ema = config["inferencer"]["inference_ops_args"]["use_ema"]
        model_key = "ema" if use_ema else "model"
        logger.info(f"Loading checkpoint from {ckpt_dir} with model_key={model_key}...")
        load_checkpoint_for_inference(ckpt_dir, self.denoiser, model_key=model_key)

        logger.info("Setting up InferenceOps...")
        # Create dataloader
        train_dataloader = None
        data_module = None
        if "data" in config:
            logger.info("Setting up data streamer...")
            data_module = DataStreamer(self.config["data"])
            train_dataloader = data_module.train_dataloader()

        inference_ops = InferenceOps(lfm=self.latent_fm, train_dataloader=train_dataloader, data_module=data_module)

        logger.info("Running inference...")
        inference_ops(**config["inferencer"]["inference_ops_args"])

        dist.destroy_process_group()
