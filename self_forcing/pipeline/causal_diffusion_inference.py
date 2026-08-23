from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = getattr(args, "sampling_steps", 50)
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        # idea #6 velocity-residual corrector r_phi (optional; None == frozen baseline)
        self.corrector = None
        self.corrector_gate = None   # optional 1-D tensor [num_train_timesteps] -> alpha(t)
        self.corrector_alpha = 1.0
        self.corrector_hist = 8      # committed-latent history window fed to r_phi

        # fractional History Guidance baseline (DFoT 2502.06764; BAgger w=1.2):
        # v = v(z|h_noised) + w*(v(z|h_clean) - v(z|h_noised)); weak branch caches context at hg_sigma
        self.hg_scale = 0.0          # 0 == off; BAgger uses 1.2
        self.hg_sigma = 0.5          # fractional history-corruption level (our-variant mode)
        self.hg_exact = False        # BAgger's exact eq.: v(uncond) + w_t[v(c)-v(uncond)] + w_hg[v(h_p800,uncond)-v(noise-ctx,uncond)]
        self.kv_cache_hg_pos = None
        self.kv_cache_hg_neg = None

        # Pathwise Test-Time Correction baseline (arXiv 2602.05871, their chosen config = steps {500, 250}):
        # at transitions into these noise levels, renoise x0_hat, denoise once under the earliest-chunk
        # reference context S0, renoise again, resume. Multi-step-host port: the reference chunk is
        # re-encoded at the adjacent rope position (their hosts re-anchor via the rolling cache).
        self.ttc_steps = None        # e.g. [500, 250]; None == off
        self.kv_cache_ttc_pos = None
        self.kv_cache_ttc_neg = None
        self._ttc_ref_latents = None

    def attach_corrector(self, corrector, gate=None, alpha=1.0):
        """v_rect = flow_pred + alpha(t) * r_phi(z_t, history, t). alpha=0 / None == baseline."""
        self.corrector = corrector
        self.corrector_gate = gate
        self.corrector_alpha = alpha

    def _cache_hg_context(self, latents, conditional_dict, unconditional_dict,
                          current_start_frame, cache_start_frame):
        """Cache weak-history branches. Our-variant: hg_sigma-noised ctx, cond+uncond.
        Exact (BAgger eq.): hg_pos <- ctx noised @ p=800 (uncond text); hg_neg <- pure-noise ctx (uncond)."""
        if self.hg_exact:
            branches = (
                (0.8, unconditional_dict, self.kv_cache_hg_pos, self.crossattn_cache_neg),
                (1.0, unconditional_dict, self.kv_cache_hg_neg, self.crossattn_cache_neg))
        else:
            branches = (
                (self.hg_sigma, conditional_dict, self.kv_cache_hg_pos, self.crossattn_cache_pos),
                (self.hg_sigma, unconditional_dict, self.kv_cache_hg_neg, self.crossattn_cache_neg))
        for s, cond, cache, xcache in branches:
            noised = (1 - s) * latents + s * torch.randn_like(latents)
            t = torch.full((latents.shape[0], latents.shape[1]), min(s * self.num_train_timesteps, 999),
                           device=latents.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=noised, conditional_dict=cond, timestep=t,
                kv_cache=cache, crossattn_cache=xcache,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length)

    def _ttc_encode_ref(self, conditional_dict, unconditional_dict, current_start_frame):
        """Rebuild the S0 reference caches: only the earliest generated chunk, re-encoded at the
        rope position directly preceding the current chunk (stays inside the trained local window)."""
        nfb = self.num_frame_per_block
        zeros_t = torch.zeros([self._ttc_ref_latents.shape[0], nfb],
                              device=self._ttc_ref_latents.device, dtype=torch.float32)
        branches = ((conditional_dict, self.kv_cache_ttc_pos, self.crossattn_cache_pos),
                    (unconditional_dict, self.kv_cache_ttc_neg, self.crossattn_cache_neg))
        for cond, cache, xcache in branches:
            for block in cache:
                # anchor the empty cache at the ref's write position: the cache write computes
                # local indices as local_end + (current_end - global_end), so a zeroed global_end
                # with a far-ahead current_start would index past the cache
                block["global_end_index"].fill_((current_start_frame - nfb) * self.frame_seq_length)
                block["local_end_index"].zero_()
            self.generator(
                noisy_image_or_video=self._ttc_ref_latents, conditional_dict=cond, timestep=zeros_t,
                kv_cache=cache, crossattn_cache=xcache,
                current_start=(current_start_frame - nfb) * self.frame_seq_length,
                cache_start=0)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        low_memory: bool = False,  # accepted+ignored (distilled pipeline uses it)
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )
        self._ttc_ref_latents = None  # per-video reference; captured from the first generated chunk

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
                self.crossattn_cache_neg[block_index]["is_init"] = False
            # reset kv cache
            hg_caches = [c for c in (self.kv_cache_hg_pos, self.kv_cache_hg_neg,
                                     self.kv_cache_ttc_pos, self.kv_cache_ttc_neg) if c is not None]
            for block_index in range(len(self.kv_cache_pos)):
                for cache in [self.kv_cache_pos, self.kv_cache_neg] + hg_caches:
                    cache[block_index]["global_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)
                    cache[block_index]["local_end_index"] = torch.tensor(
                        [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                if self.hg_scale:
                    self._cache_hg_context(current_ref_latents, conditional_dict, unconditional_dict,
                                           current_start_frame, cache_start_frame)
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)

            # Chunk overlap+blend (seam-pulse fix a): extend the window one latent back and pin
            # that slot to the committed frame at the CURRENT noise level each step, so new frames
            # self-attend to the boundary at matched noise instead of jumping to clean context.
            # The KV write path is position-derived, so the earlier window start overwrites the
            # boundary frame's cache slot in place; the t=0 recache below restores it.
            ov_mode = int(getattr(self, "overlap_blend", 0) or 0)
            ov = 1 if ov_mode and cache_start_frame > 0 else 0
            if ov:
                assert not self.hg_scale and not self.ttc_steps and self.corrector is None \
                    and getattr(self, "freq_lambda", None) is None, "overlap_blend: unsupported combo"
                x0_prev = output[:, cache_start_frame - 1:cache_start_frame]
                eps_ov = torch.randn_like(x0_prev)
                sig0 = sample_scheduler.sigmas[0].to(device=latents.device, dtype=latents.dtype)
                latents = torch.cat([(1 - sig0) * x0_prev + sig0 * eps_ov, latents], dim=1)
            # Pathwise TTC: mark the loop iterations whose NEXT timestep is a correction level
            ttc_iters = set()
            if self.ttc_steps and self._ttc_ref_latents is not None:
                ts = sample_scheduler.timesteps
                for target in self.ttc_steps:
                    inext = int(torch.argmin((ts.float() - target).abs()).item())
                    if inext >= 1:
                        ttc_iters.add(inext - 1)
            for step_i, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, ov + current_num_frames], device=noise.device, dtype=torch.float32
                )

                # idea #1 Drift-SNR gate for the LoRA corrector: per-timestep alpha via lora scale
                if getattr(self, "lora_gate", None) is not None:
                    from wan.modules.lora import set_lora_scale
                    ti = torch.argmin((self.lora_gate["timesteps"].to(t.device) - t).abs()).item()
                    set_lora_scale(self.generator.model, float(self.lora_gate["alpha"][ti]))

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=(current_start_frame - ov) * self.frame_seq_length,
                    cache_start=(cache_start_frame - ov) * self.frame_seq_length
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=(current_start_frame - ov) * self.frame_seq_length,
                    cache_start=(cache_start_frame - ov) * self.frame_seq_length
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                # fractional History Guidance: extrapolate clean-history vs noised-history branches
                if self.hg_scale:
                    hg_cond, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        conditional_dict=unconditional_dict if self.hg_exact else conditional_dict,
                        timestep=timestep, kv_cache=self.kv_cache_hg_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length)
                    hg_uncond, _ = self.generator(
                        noisy_image_or_video=latent_model_input, conditional_dict=unconditional_dict,
                        timestep=timestep, kv_cache=self.kv_cache_hg_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length)
                    if self.hg_exact:
                        # hg_cond ran on hg_pos (h_p800, uncond ctx-branch), hg_uncond on hg_neg (noise ctx)
                        flow_pred = flow_pred + self.hg_scale * (hg_cond - hg_uncond)
                    else:
                        flow_weak = hg_uncond + self.args.guidance_scale * (hg_cond - hg_uncond)
                        flow_pred = flow_weak + self.hg_scale * (flow_pred - flow_weak)

                # v3 candidate: frequency-selective correction — full low-freq (drift lives there,
                # Gate-B), damped high-freq/structure (progression lives there). Needs LoRA applied;
                # computes correction explicitly via a second pass at scale 0 (2x NFE, experiment only).
                if getattr(self, "freq_lambda", None) is not None:
                    from wan.modules.lora import set_lora_scale
                    set_lora_scale(self.generator.model, 0.0)
                    base_cond, _ = self.generator(
                        noisy_image_or_video=latent_model_input, conditional_dict=conditional_dict,
                        timestep=timestep, kv_cache=self.kv_cache_pos,
                        crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length)
                    base_unc, _ = self.generator(
                        noisy_image_or_video=latent_model_input, conditional_dict=unconditional_dict,
                        timestep=timestep, kv_cache=self.kv_cache_neg,
                        crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length)
                    set_lora_scale(self.generator.model, 1.0)
                    v_base = base_unc + self.args.guidance_scale * (base_cond - base_unc)
                    dv = (flow_pred - v_base).float()
                    F2 = torch.fft.rfft2(dv, dim=(-2, -1))
                    H, Wd = dv.shape[-2], dv.shape[-1] // 2 + 1
                    fy = torch.fft.fftfreq(H, device=dv.device).abs().view(-1, 1)
                    fx = torch.fft.rfftfreq(dv.shape[-1], device=dv.device).view(1, -1)
                    lowmask = ((fy ** 2 + fx ** 2).sqrt() < self.freq_cutoff).to(F2.dtype)
                    dv_low = torch.fft.irfft2(F2 * lowmask, s=dv.shape[-2:], dim=(-2, -1))
                    dv_high = dv - dv_low
                    flow_pred = (v_base.float() + dv_low + self.freq_lambda * dv_high).to(flow_pred.dtype)

                # idea #6: v_rect = v_theta + alpha(t) * r_phi(z_t, history, t)
                if self.corrector is not None:
                    history = output[:, max(0, current_start_frame - self.corrector_hist):current_start_frame]
                    res = self.corrector(latent_model_input, timestep, history=history if history.shape[1] else None)
                    a = self.corrector_alpha
                    if self.corrector_gate is not None:
                        g = self.corrector_gate.to(res.device)[timestep.long().clamp(0, self.corrector_gate.numel() - 1)]
                        a = a * g[:, :, None, None, None]  # (B,F,1,1,1)
                    flow_pred = flow_pred + a * res

                if step_i in ttc_iters:
                    # Pathwise TTC sandwich (Alg. 1 of arXiv 2602.05871), replacing this solver step:
                    # x0_hat -> Psi to next level -> denoise under S0 -> x0_c -> Psi again -> resume S_t
                    sig = sample_scheduler.sigmas[step_i].to(latents)
                    sig_c = sample_scheduler.sigmas[step_i + 1].to(latents)
                    x0_hat = latents - sig * flow_pred
                    t_c = sample_scheduler.timesteps[step_i + 1] * torch.ones_like(timestep)
                    z_c = (1 - sig_c) * x0_hat + sig_c * torch.randn_like(x0_hat)
                    self._ttc_encode_ref(conditional_dict, unconditional_dict, current_start_frame)
                    v_cond, _ = self.generator(
                        noisy_image_or_video=z_c, conditional_dict=conditional_dict, timestep=t_c,
                        kv_cache=self.kv_cache_ttc_pos, crossattn_cache=self.crossattn_cache_pos,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length)
                    v_unc, _ = self.generator(
                        noisy_image_or_video=z_c, conditional_dict=unconditional_dict, timestep=t_c,
                        kv_cache=self.kv_cache_ttc_neg, crossattn_cache=self.crossattn_cache_neg,
                        current_start=current_start_frame * self.frame_seq_length,
                        cache_start=cache_start_frame * self.frame_seq_length)
                    v_c = v_unc + self.args.guidance_scale * (v_cond - v_unc)
                    x0_c = z_c - sig_c * v_c
                    latents = (1 - sig_c) * x0_c + sig_c * torch.randn_like(x0_c)
                    # advance the solver past this step and clear its multistep history
                    # (the injected state breaks the lower-order continuity assumption)
                    sample_scheduler._step_index = step_i + 1
                    sample_scheduler.model_outputs = [None] * len(sample_scheduler.model_outputs)
                    sample_scheduler.lower_order_nums = 0
                    sample_scheduler.last_sample = None  # also disarm the UniC corrector update
                    continue

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                if ov and ov_mode == 1:  # v1: pin the boundary slot to the committed frame
                    sig_n = sample_scheduler.sigmas[step_i + 1].to(latents) \
                        if step_i + 1 < len(sample_scheduler.sigmas) else torch.zeros((), device=latents.device, dtype=latents.dtype)
                    latents[:, :1] = (1 - sig_n) * x0_prev + sig_n * eps_ov

            if ov:
                if ov_mode == 2:  # v2: boundary frame regenerated freely; cross-fade with committed
                    output[:, cache_start_frame - 1:cache_start_frame] = 0.5 * x0_prev + 0.5 * latents[:, :1]
                latents = latents[:, ov:]

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents
            if self.ttc_steps is not None and self._ttc_ref_latents is None:
                self._ttc_ref_latents = latents.detach().clone()  # earliest chunk = S0 reference

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            # (DF noisy-context baseline, BAgger sigma_test: cache noised context at matching t)
            ctx_latents, ctx_timestep = latents, timestep * 0
            if ov:  # restore the boundary frame's clean KV (blended frame in v2) alongside the new frames
                ctx_latents = torch.cat([output[:, cache_start_frame - 1:cache_start_frame], latents], dim=1)
            if getattr(self, "context_noise_sigma", 0):
                s = self.context_noise_sigma
                ctx_latents = (1 - s) * ctx_latents + s * torch.randn_like(ctx_latents)
                ctx_timestep = torch.ones_like(timestep) * (s * self.num_train_timesteps)
            self.generator(
                noisy_image_or_video=ctx_latents,
                conditional_dict=conditional_dict,
                timestep=ctx_timestep,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=(current_start_frame - ov) * self.frame_seq_length,
                cache_start=(cache_start_frame - ov) * self.frame_seq_length
            )
            self.generator(
                noisy_image_or_video=ctx_latents,
                conditional_dict=unconditional_dict,
                timestep=ctx_timestep,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=(current_start_frame - ov) * self.frame_seq_length,
                cache_start=(cache_start_frame - ov) * self.frame_seq_length
            )

            if self.hg_scale:
                self._cache_hg_context(latents, conditional_dict, unconditional_dict,
                                       current_start_frame, cache_start_frame)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

        if getattr(self, "hg_scale", 0) or getattr(self, "ttc_steps", None):
            def _mk():
                return [{"k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                         "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                         "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                         "local_end_index": torch.tensor([0], dtype=torch.long, device=device)}
                        for _ in range(self.num_transformer_blocks)]
            if getattr(self, "hg_scale", 0):
                self.kv_cache_hg_pos = _mk()
                self.kv_cache_hg_neg = _mk()
            if getattr(self, "ttc_steps", None):
                self.kv_cache_ttc_pos = _mk()
                self.kv_cache_ttc_neg = _mk()

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos  # always store the clean cache
        self.crossattn_cache_neg = crossattn_cache_neg  # always store the clean cache

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
