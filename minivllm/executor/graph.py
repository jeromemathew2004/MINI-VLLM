import torch
from tqdm import tqdm
from minivllm.config.config import Config
from minivllm.executor.context import Context


class CudaGraphRunner:
    """Pre-captured graphs for decode-shaped forward passes.

    "Decode-shaped" means one query row per batch entry, each attending to its
    own prefix through `cache_seqlens` and `block_table`. Plain decoding puts one
    row per *request* there, but that is a convention of the caller, not of this
    class: a speculative verify pass puts K+1 rows per request and a draft
    proposal step puts one or two, and all three replay the same graphs. See
    `Executor.verify` for why the multi-token shape is expressible this way.

    Graphs exist because kernel launch overhead dominates decode at small batch
    sizes. Measured on the dev host, a graphed step is ~8x faster than the eager
    one for Qwen3-0.6B at batch 1 (`experiments/spec_breakeven.py`), which is
    most of what makes speculative decoding viable here at all.
    """

    def __init__(self, model, config: Config, max_batch_size: int,
                 hf_config=None, label: str = "model"):
        self.model = model
        self.config = config
        self.label = label
        hf_config = hf_config if hf_config is not None else config.hf_config
        self.max_batch_size = min(max_batch_size, 256)

        # Two sizing bugs fixed here, both reachable from plain decode:
        #   - the small sizes were unconditional, so max_batch_size < 8 captured
        #     a graph wider than the buffers backing it, silently truncating;
        #   - the largest captured size was max_num_batched_seqs only when that
        #     happened to be 1/2/4/8 or a multiple of 16. At 20, replay()'s
        #     `next(b for b in batch_size_list if b >= bs)` raised StopIteration
        #     on a full batch.
        sizes = {b for b in (1, 2, 4, 8) if b <= self.max_batch_size}
        sizes.update(range(16, self.max_batch_size + 1, 16))
        sizes.add(self.max_batch_size)
        self.batch_size_list = sorted(sizes)

        self.graphs = {}
        self.pool = None

        self.input_ids = torch.zeros(self.max_batch_size, dtype=torch.int64, device='cuda')
        self.positions = torch.zeros(self.max_batch_size, dtype=torch.int64, device='cuda')
        self.slot_mapping = torch.zeros(self.max_batch_size, dtype=torch.int32, device='cuda')
        self.cache_seqlens = torch.zeros(self.max_batch_size, dtype=torch.int32, device='cuda')
        self.block_table = torch.zeros((self.max_batch_size, config.kv_cache_num_blocks), dtype=torch.int32, device='cuda')
        self.outputs = torch.zeros(self.max_batch_size, hf_config.vocab_size, device='cuda')

    def can_replay(self, batch_size: int) -> bool:
        return batch_size <= self.max_batch_size

    @torch.inference_mode()
    def capture(self):
        pbar = tqdm(
            reversed(self.batch_size_list),
            desc=f"Capturing CUDA graphs ({self.label})...",
        )

        for batch_size in pbar:
            pbar.set_postfix({
                "Batch Size": batch_size,
            })
            self._capture_batch(batch_size)
        pbar.close()

    def _capture_batch(self, batch_size: int):
        ctx = Context(
            prefill=False,
            positions=self.positions[:batch_size],
            slot_mapping=self.slot_mapping[:batch_size],
            cache_seqlens=self.cache_seqlens[:batch_size],
            block_table=self.block_table[:batch_size],
        )

        # we must run the model once before capturing the graph, since some pytorch ops need compile.
        self.outputs[:batch_size] = self.model(
            ctx, self.input_ids[:batch_size], self.positions[:batch_size])

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self.outputs[:batch_size] = self.model(ctx, self.input_ids[:batch_size], self.positions[:batch_size])

        # if this graph uses a new memory pool, we save it for next graphs.
        self.pool = g.pool()

        self.graphs[batch_size] = g
        torch.cuda.synchronize()

    @torch.inference_mode()
    def replay(self, ctx: Context, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the captured graph for `input_ids`, padding up to a captured size.

        Padded rows are made inert rather than merely ignored: `slot_mapping`
        of -1 makes `store_kvcache` skip them, and `cache_seqlens` of 0 leaves
        them attending to nothing. Their outputs are garbage and are never
        returned.

        The returned tensor is a **view of this runner's persistent output
        buffer**, so the caller must consume or copy it before this runner
        replays again. Separate runners (target and draft) have separate
        buffers and do not interfere.
        """
        bs = input_ids.size(0)
        graph = self.graphs[next(b for b in self.batch_size_list if b >= bs)]
        self.input_ids[:bs] = input_ids
        self.positions[:bs] = ctx.positions
        self.slot_mapping.fill_(-1)
        self.slot_mapping[:bs] = ctx.slot_mapping
        self.cache_seqlens.zero_()
        self.cache_seqlens[:bs] = ctx.cache_seqlens
        self.block_table[:bs, :ctx.block_table.size(1)] = ctx.block_table
        graph.replay()
        return self.outputs[:bs]
