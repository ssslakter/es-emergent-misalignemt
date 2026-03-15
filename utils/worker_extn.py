import gc
import time
import torch


def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """
    Methods injected into vLLM worker processes by the ES trainer.

    Weight mutation API
    -------------------
    - perturb_self_weights(seed, scale)
        Add scale * N(seed) to all parameters. Records (seed, scale) so
        restore_self_weights() needs no arguments.
    - restore_self_weights()
        Undo the last perturb_self_weights call exactly.
    - apply_update(perturbations: list[tuple[int, float]])
        Single-pass ES update: accumulate sum_i coeff_i * N(seed_i) and add
        to weights. Much faster than calling perturb 30 times.

    Inter-engine communication
    --------------------------
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)

    Persistence
    -----------
    - save_self_weights_to_disk(filepath)
    - load_self_weights_from_disk(filepath)
    """

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _noise_for_param(self, p, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=p.device)
        gen.manual_seed(int(seed))
        return torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Weight mutation
    # ------------------------------------------------------------------ #

    def perturb_self_weights(self, seed: int, scale: float) -> bool:
        """Add scale * N(seed) to all parameters. Records the call for restore."""
        scale = float(scale)
        for _, p in self.model_runner.model.named_parameters():
            noise = self._noise_for_param(p, seed)
            p.data.add_(scale * noise)
            del noise
        self._sync()
        # Store so restore() can invert without the caller tracking state
        self._last_perturb = (int(seed), scale)
        return True

    def restore_self_weights(self) -> bool:
        """Invert the last perturb_self_weights call exactly."""
        if not hasattr(self, "_last_perturb"):
            raise RuntimeError("restore called before any perturb")
        seed, scale = self._last_perturb
        for _, p in self.model_runner.model.named_parameters():
            noise = self._noise_for_param(p, seed)
            p.data.add_(-scale * noise)
            del noise
        self._sync()
        del self._last_perturb
        return True

    def apply_update(self, perturbations: list) -> bool:
        """
        Single-pass ES weight update.

        perturbations: list of (seed, coeff) pairs
            coeff = (alpha / pop_size) * normalised_reward

        Accumulates the full weighted sum in one sweep over parameters
        instead of calling perturb N times.
        """
        for _, p in self.model_runner.model.named_parameters():
            delta = torch.zeros_like(p.data)
            for seed, coeff in perturbations:
                noise = self._noise_for_param(p, seed)
                delta.add_(float(coeff) * noise)
                del noise
            p.data.add_(delta)
            del delta
        self._sync()
        return True

    # ------------------------------------------------------------------ #
    # Inter-engine communication
    # ------------------------------------------------------------------ #

    def init_inter_engine_group(
        self, master_address: str, master_port: int, rank: int, world_size: int
    ) -> bool:
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int) -> bool:
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        self._sync()
        return True

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save_self_weights_to_disk(self, filepath: str) -> bool:
        state_dict = {
            name: p.detach().cpu()
            for name, p in self.model_runner.model.named_parameters()
        }
        torch.save(state_dict, filepath)
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True

    def load_self_weights_from_disk(self, filepath: str) -> bool:
        state_dict = torch.load(filepath, map_location=self.device)
        for name, p in self.model_runner.model.named_parameters():
            if name in state_dict:
                p.data.copy_(state_dict[name].to(p.device))
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
