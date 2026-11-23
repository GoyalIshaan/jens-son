# LLM Mathematics and Hardware Architecture Factbook

Research date: October 21, 2026. This reference collects computation, hardware mechanisms, published implementations, and company disclosures relevant to an LLM chip project. Architecture selection remains open.

The research used 41 targeted Exa searches returning 205 result entries, followed by reads of selected papers, repositories, and vendor documentation. Those entries include duplicate papers, multiple versions, and secondary discovery results; they are not 205 independent studies. The factual citations below point to primary sources. The accompanying search provenance file records discovery separately from evidence used here.

Equations and numerical examples labeled **derived** are calculations from stated assumptions, not hardware measurements. A vendor specification is a specification; a vendor benchmark remains a vendor benchmark. Synthesis, FPGA implementation, simulation, and fabricated silicon establish different things. Live documentation was inspected on the research date; dated announcements establish only what was announced at that date.

## 1. The computation inside a language model

### 1.1 Tokens, weights, activations, and inference state

A token is a discrete vocabulary index. An embedding lookup maps that index to a vector. Each model layer transforms vectors using learned weights and intermediate activations. An output projection produces vocabulary logits, and a decoding rule selects the next token. Autoregressive generation then repeats the forward pass using that token and the saved inference state. The original Transformer paper establishes the attention architecture; later inference work explains the different execution regimes. [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102).

These objects have different lifetimes:

| Object | Lifetime and access pattern | Hardware consequence |
| --- | --- | --- |
| Model weights | Usually fixed across many requests; accessed layer by layer | Compression, reuse across tokens, caching, or permanent placement can reduce transfers |
| Activations | Produced and consumed during a forward pass | Fusion and local buffers can eliminate intermediate transfers |
| Transformer KV cache | Separate state for each request; grows with retained context | Capacity, bandwidth, layout, paging, and sharing matter |
| Recurrent state | Updated each token; fixed shape for a fixed pure recurrent model | Local state placement and update dependencies matter |
| Logits and sampling buffers | Vocabulary-sized outputs; multiple positions in some decoding algorithms | Reductions, top-k, random sampling, and temporary storage can matter |

The recurrent-state distinction follows from Mamba-2 and RWKV; the logits/sampling issue is explicitly studied in diffusion hardware work. [Mamba-2](https://arxiv.org/abs/2405.21060), [RWKV-7](https://arxiv.org/abs/2503.14456), [Diffusion sampling NPU paper](https://arxiv.org/html/2601.20706v1).

### 1.2 A conventional decoder layer

Use the following notation:

| Symbol | Meaning |
| --- | --- |
| B | Number of sequences processed together |
| T | Prompt length in prefill, or retained context length in decode |
| d | Model hidden dimension |
| h_q | Number of query heads |
| h_kv | Number of key/value heads |
| d_h | Dimension of each attention head |
| m | MLP intermediate dimension |
| L | Number of layers |
| b_w, b_kv | Stored bytes per weight or KV element, including any overhead being modeled |

For the familiar case h_q d_h = d, the attention projections are:

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
\]

In prefill, X has B times T rows. In a normal one-token decode step, it has B rows. Per query head, attention computes

\[
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
\]

where M applies the causal mask and possibly another model-specific mask. The concatenated heads pass through an output projection. With GQA, several query heads share a key/value head. Some models use an attention projection width different from d, so actual shapes must come from the model configuration. [Transformer](https://arxiv.org/abs/1706.03762), [GQA](https://arxiv.org/abs/2305.13245).

A common modern MLP uses SwiGLU:

\[
\operatorname{MLP}(x)=\left[\operatorname{SiLU}(xW_g)\odot(xW_u)\right]W_d,
\qquad \operatorname{SiLU}(z)=z\,\sigma(z).
\]

It has two up projections and one down projection. A GPT-2-style GELU MLP instead uses two large projection matrices. This changes both parameter count and arithmetic. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).

Residual additions, normalization, position handling, and sampling are also part of the model. GPT-2 uses learned positional embeddings and LayerNorm; RoPE and RMSNorm are common in later architectures. RMSNorm computes a vector reduction, reciprocal square root, and scaling:

\[
y_i=\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_j x_j^2+\epsilon}}.
\]

RoPE rotates pairs of query/key coordinates using position-dependent sine/cosine values. It can use precomputed coefficients; a chip does not have to evaluate fresh trigonometric functions for every coordinate. [RMSNorm](https://arxiv.org/abs/1910.07467), [RoFormer](https://arxiv.org/abs/2104.09864).

### 1.3 Arithmetic counts

**Derived, counting a multiplication and addition as two operations.** Multiplying an M by K matrix by a K by N matrix requires approximately 2MKN operations. If weights are integers, the same count describes arithmetic operations rather than floating-point operations.

Under h_q d_h = d, and ignoring biases and normalization parameters, the attention projection parameter count per block is

\[
P_{\mathrm{attn\ proj}}=2d^2+2d h_{kv}d_h.
\]

The SwiGLU MLP contributes

\[
P_{\mathrm{MLP}}=3dm.
\]

The corresponding dense linear arithmetic is roughly twice these counts per token. These expressions are a derivation from the projection shapes, not a universal count for every architecture.

For an illustrative block with d = 4096, m = 14336, h_q = 32, h_kv = 8, d_h = 128:

| Component | Parameters, derived |
| --- | ---: |
| Attention projections | 41,943,040 |
| SwiGLU MLP | 176,160,768 |
| MLP fraction of these block matrices | 80.77% |

The often-repeated statement that the MLP is always two-thirds of a Transformer is therefore not a general identity. The fraction depends on MLP width, gating, GQA, and other architectural choices.

For prefill attention itself, calculating all entries of QK-transpose and multiplying by V costs approximately 4BT²d operations per layer. If an implementation skips the causal upper triangle, the dominant count is about 2BT(T+1)d. A single-token decode step costs about 4BTd for these two attention products. GQA reduces stored KV and K/V projection widths, but the attention dot products still serve all query heads. [GQA](https://arxiv.org/abs/2305.13245), [FlashAttention](https://arxiv.org/abs/2205.14135).

For a large dense model, the linear-layer inference rule of thumb is roughly 2N operations per token, where N counts the participating matrix weights. Embedding lookups, attention interactions with context, nonlinear operations, padding, and precision conversion need separate treatment. The familiar training estimate near 6ND for N parameters and D training tokens includes forward and backward matrix work; it is not a decode latency model. The factor can also change with recomputation and architecture. [Inference analysis](https://arxiv.org/abs/2211.05102), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

## 2. Prefill, decode, and the memory limits

### 2.1 They are different matrix shapes

| Phase | Linear-layer shape | Main opportunity | Possible limiting resource |
| --- | --- | --- | --- |
| Prefill | Many prompt-token rows times weights | Reuse each weight across many rows | Matrix throughput, attention work, local memory, communication |
| Single-request decode | One row times weights | Low latency for one dependency chain | Weight/KV transfers, reductions, synchronization |
| Batched decode | Multiple requests' rows times weights | Share a weight read across requests | Matrix throughput, KV bandwidth/capacity, expert routing |
| Speculative verification | Several proposed positions evaluated together | Amortize a target-model pass | Verification work, draft cost, acceptance, state handling |

Prefill is often more compute-intensive, and small-batch decode often more bandwidth-intensive. These are workload-dependent tendencies. A very long prompt, long KV cache, small model, irregular experts, or narrow matrix can alter the bottleneck. [Pope et al.](https://arxiv.org/abs/2211.05102), [DistServe](https://arxiv.org/html/2401.09670v3), [Speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html).

### 2.2 Arithmetic intensity and the roofline

Arithmetic intensity is operations divided by bytes transferred at a specified memory boundary. For a matrix-vector multiply whose dominant transfer is a weight matrix stored at b_w bytes per weight, the ideal intensity is about 2/b_w. With B vectors reusing each loaded weight, it is about 2B/b_w. Activations, partial sums, KV, scales, and reloads reduce that simplified estimate.

**Derived examples:** FP16 weights give approximately B operations per weight byte; INT8 gives 2B; raw 4-bit gives 4B. A lower bit width improves the bandwidth term only when packing, metadata, conversion, and the arithmetic implementation permit it.

For a compute engine capable of C operations per second and memory bandwidth BW bytes per second, the roofline bound is

\[
\text{achievable rate}\le\min(C,\;BW\times\text{arithmetic intensity}).
\]

The latency lower bound is correspondingly

\[
t\ge\max(F/C,\;D/BW).
\]

This ideal bound assumes overlap. Real systems may pay serial transfer/computation stages plus launch, synchronization, bank-conflict, and communication costs. The memory boundary must be specified: SRAM traffic and HBM traffic have different bandwidths and energy costs. [Inference performance model](https://arxiv.org/abs/2211.05102), [FlashAttention IO analysis](https://arxiv.org/abs/2205.14135).

### 2.3 Weight-storage example

**Derived for exactly 8 billion weights, using decimal GB; excludes scales, padding, metadata, nonquantized tensors, cache, and activations.**

| Representation | Raw weight storage |
| --- | ---: |
| FP16 or BF16 | 16 GB |
| INT8 or FP8 | 8 GB |
| 4-bit | 4 GB |
| Plain 2-bit storage | 2 GB |
| Ideal fixed-rate ternary information limit, log2(3) bits | About 1.585 GB |

Ternary weights are three-valued, not binary. Practical encoding can use 2 bits per value or pack multiple values: five ternary weights have 3⁵ = 243 possibilities and fit into 8 bits, giving 1.6 bits per weight. Decoding that representation costs hardware. [BitNet b1.58](https://arxiv.org/abs/2402.17764), [TerEffic encoding](https://arxiv.org/html/2502.16473v2).

**Derived bandwidth bound:** if all participating weights of that 8B model were streamed once per token at an ideal 1 TB/s, FP16 weights alone would require 16 ms/token, or at most 62.5 tokens/s. Raw 4-bit weights alone would require 4 ms/token, or at most 250 tokens/s. Neither number is a prediction: KV access, computation, metadata, imperfect utilization, and communication remain. Model placement in on-chip memory changes the assumed boundary.

### 2.4 KV cache can be large even when weights are compressed

For ordinary GQA with fixed retained context length,

\[
\boxed{\text{KV bytes}=2BLT h_{kv}d_h b_{kv}}.
\]

The factor of two accounts for keys and values. If layers have different attention patterns, sum their retained cache sizes individually. MLA and pure recurrent models require different formulas.

**Derived illustrative example:** L = 32, h_kv = 8, d_h = 128, and 2-byte KV elements gives 128 KiB of cache per retained token per request across all layers.

| Retained context | Cache for one request |
| --- | ---: |
| 8,192 tokens | 1 GiB |
| 16,384 tokens | 2 GiB |
| 131,072 tokens | 16 GiB |

Sixty-four independent 8K requests therefore require 64 GiB just for that KV representation. GQA gives a fourfold cache reduction relative to 32 independent KV heads in this example, while still maintaining 32 query heads. A good kernel also reuses shared K/V across the query heads; materializing repeated K/V can squander that bandwidth benefit. [GQA](https://arxiv.org/abs/2305.13245), [KIVI](https://arxiv.org/abs/2402.02750).

Weight reuse improves with batching; independent requests' KV caches usually do not share the same property. Under ideal weight reuse, cache traffic grows with B while a weight scan is amortized across B requests. This explains why the dominant bottleneck can shift as batch size and context increase. Shared prefixes provide a special case of cache reuse. [PagedAttention](https://arxiv.org/html/2309.06180).

**Derived KV-only attention intensity:** the two attention products use about 4BT h_q d_h operations and read about 2BT h_kv d_h b_kv bytes if shared KV is read only once. Their ratio is 2h_q/(h_kv b_kv). For 32 query heads, eight KV heads, and 2-byte KV, this is only four operations per KV byte. Batching independent requests does not improve this ratio the way it improves reuse of common model weights. This simplified expression excludes output transfers, cache writes, softmax, and other work.

### 2.5 Cache techniques change different things

| Technique | What it changes | What still needs accounting |
| --- | --- | --- |
| GQA/MQA | Number of independent K/V heads | Query work and model compatibility |
| KV quantization | Bytes per stored element | Scales, conversion, quality, residual high-precision storage |
| PagedAttention | Allocation, fragmentation, and sharing | Attention still reads the retained context |
| Prefix caching | Reuse of compatible common-prefix state | Correct model, positional, adapter, and cache identity |
| Sliding window | Retained tokens in layers trained to use a window | Global-attention layers and model semantics |
| MLA | Store a compressed latent representation | Attention algorithm, projection work, positional component |
| Sparse attention | Which positions are attended | Selection cost, gathers, approximation or model design |

PagedAttention is a memory-management/data-layout technique, while FlashAttention changes how attention is evaluated. They address different costs and can coexist. [PagedAttention](https://arxiv.org/html/2309.06180), [FlashAttention](https://arxiv.org/abs/2205.14135).

## 3. Hardware mechanisms behind those costs

### 3.1 Matrix engines and dataflows

A systolic array passes operands between processing elements while multiplying and accumulating. Its efficiency comes from local reuse and regular communication. Weight-stationary dataflow keeps weights near the arithmetic while activations and sums move. Output-stationary dataflow keeps partial sums local while operands move. These choices change SRAM accesses, accumulator placement, broadcast needs, and reuse. Gemmini implements both and documents banked scratchpad SRAM, DMA, and an accumulator memory. [Gemmini](https://github.com/ucb-bar/gemmini).

A square array is not automatically well utilized by a single-row decode matrix. Depending on its mapping, rows or columns can sit idle. Designs can partition arrays, process multiple requests, map more reduction work across units, or use vector/dot-product engines. Those changes have tradeoffs in wiring, reduction, buffers, and scheduling. MatX explicitly discloses a splittable systolic array for smaller matrices; Positron describes a reconfigurable GEMM/GEMV array. These are published implementation choices, not evidence that a single geometry is best. [MatX One](https://matx.com/research/series_b), [Asimov](https://www.positron.ai/asimov).

**Derived sizing illustration:** an ideal R by C array issuing one MAC per PE per cycle has peak 2RC operations/cycle. A 16 by 16 array at an assumed 1 GHz has an arithmetic ceiling of 512 billion operations/s. This does not establish attainable frequency or throughput. Feeding 16 new INT8 values on each of two array edges each cycle would itself require 32 GB/s at those edges under that particular streaming mapping, before accounting for output traffic and reuse elsewhere.

### 3.2 Memory hierarchy and the network are part of the processor

Registers, local SRAM, shared SRAM, HBM, conventional DRAM, host memory, and flash trade capacity, cost, bandwidth, and access energy. Local memory is useful only when it can supply the required accesses. Capacity alone does not specify the number of banks or ports, multicast ability, conflict behavior, or physical wire delay. Eyeriss provides an established example of designing dataflow to reduce movement at multiple levels, although its evaluated workload is CNNs. Its exact energy numbers should not be transplanted into a modern LLM process node. [Eyeriss](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf).

Double buffering overlaps fetching the next tile with computing the current one. Fusion keeps intermediates local, but may increase live state and consume scratchpad capacity. Sharding can improve locality while making some accesses or collectives more expensive. Tenstorrent's public guide exposes separate reader, compute, and writer kernels coordinated by circular buffers in local SRAM, with native 32 by 32 tiles. [TT-Metalium guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md).

For a transfer, a useful simple model is latency = startup cost + bytes/bandwidth. With small tensors sent frequently, startup and synchronization can dominate even when link bandwidth is high. NVIDIA's Groq 3 LPX discussion explicitly targets this with compiler-scheduled communication and fine-grained overlap. [LPX architecture discussion](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/).

### 3.3 Nonlinear and vector units can limit the full pipeline

Normalization requires reductions and reciprocal square root. Softmax requires maxima, exponentials, sums, and normalization. SwiGLU requires an activation and elementwise multiplication. RoPE uses pairwise operations and coefficient access. Quantization requires scale computation or lookup, rounding, clipping, and sometimes packing/unpacking. These operations can dominate latency after matrix multiplication is accelerated, despite representing fewer total operations. FlashAttention-4 explicitly studies this imbalance on Blackwell, including exponential and shared-memory bottlenecks. [FlashAttention-4](https://arxiv.org/html/2603.05451).

Supported precision must therefore be described per operation and tensor, not only as a headline such as “INT8 model” or “FP4 chip.” A design can use 4-bit weights, 8- or 16-bit activations, higher-precision accumulation, and a different precision for norms and state.

## 4. Attention architectures and execution

### 4.1 FlashAttention and online softmax

Naive attention materializes an attention-score matrix that grows quadratically with context length. FlashAttention tiles Q, K, and V and keeps running softmax statistics, avoiding writing the full matrix to HBM. It computes the same dense attention function up to floating-point numerical differences; it does not eliminate the quadratic dense-attention arithmetic. [FlashAttention](https://arxiv.org/abs/2205.14135).

**Derived expression of the online update:** for one query, maintain the running maximum m, exponential sum ell, and unnormalized weighted-value vector z. For a new block of scores s_i and values v_i,

\[
m'=\max(m,\max_i s_i),\quad a=e^{m-m'},
\]

\[
\ell'=a\ell+\sum_i e^{s_i-m'},\qquad
z'=az+\sum_i e^{s_i-m'}v_i.
\]

The output is z/ell. This requires local state, reductions, exponentials, rescaling, and matrix products. It can be built using several hardware organizations; it is an algorithm/dataflow, not exclusively a GPU feature.

FlashAttention-2 improves parallelism and work partitioning. FlashAttention-3 exploits Hopper's asynchronous tensor computation and transfers, including FP8 techniques. FlashAttention-4, published in March 2026, targets Blackwell's changed balance of matrix throughput, shared-memory traffic, and exponential work. It includes redesigned pipelines, tensor-memory use, and softmax-related optimizations. [FlashAttention-2](https://princeton-nlp.github.io/flash-atttention-2/), [FlashAttention-3](https://arxiv.org/html/2407.08608v2), [FlashAttention-4](https://arxiv.org/html/2603.05451).

### 4.2 MLA changes the cached representation

DeepSeek-V3 uses Multi-head Latent Attention. Rather than caching separate full-dimensional K and V for every head, it caches a low-rank shared KV latent plus a separate positional key component. Its report specifies a 512-dimensional KV latent and 64-dimensional decoupled RoPE key: 576 stored scalar values per layer per token before precision and metadata. This is a representation change, not just a more compact allocator. [DeepSeek-V3 report, architecture and hyperparameters](https://arxiv.org/html/2412.19437v2).

The implementation must preserve the compression advantage through the computation. Projection fusion/absorption and the choice between expanded and latent attention paths affect bandwidth and arithmetic. Treating MLA as ordinary MHA with a different cache byte count misses part of the workload. The official inference code is a concrete reference for its supported paths. [DeepSeek-V3 inference implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py).

### 4.3 Sparse and compressed attention

Sparsity can remove attention interactions, but it introduces a selection/access problem. A regular block pattern can stream predictably. Input-dependent selection may require an indexer, top-k or related selection, gathers, metadata, and load balancing. A reduction in arithmetic does not automatically become a proportional reduction in runtime.

DeepSeek's V3.2-Exp model card introduces DeepSeek Sparse Attention. The 2026 V4 report describes Compressed Sparse Attention, which compresses along the sequence dimension before sparse selection, and Heavily Compressed Attention, which uses stronger compression with dense attention over that representation. Its published models include V4-Pro at 1.6T total / 49B active parameters and V4-Flash at 284B total / 13B active parameters. These are concrete model families whose cache and attention execution differ from a conventional GPT-2 block. [V3.2-Exp official model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/README.md), [V4 technical report](https://arxiv.org/html/2606.19348), [V4 preview release, April 24, 2026](https://api-docs.deepseek.com/news/news260424/).

The V4 report's relative FLOP/cache numbers are model-specific estimates under stated long-context settings, not measured chip speedups. Changing the attention algorithm after training is also not generally a semantics-preserving optimization.

## 5. Low precision, ternary arithmetic, and sparsity

### 5.1 Quantization is both a numerical and hardware format

For symmetric quantization, a simplified representation is x approximately equals s_x q_x. A linear output can then be written

\[
y\approx s_xs_w\sum_i q_{x,i}q_{w,i}.
\]

Returning to an output integer format requires scaling by the output scale, rounding, and clipping. Per-channel and per-group scales change the implementation. With group-dependent scales, partial sums from different groups cannot all be treated as one unscaled accumulator.

The choices include weight-only quantization such as W4A16; joint weight/activation quantization such as W8A8; FP8/FP4 formats; block-scaled or microscaled formats; and separately quantized KV. Each affects packing, multipliers, accumulation, conversion, and accuracy differently. GPTQ uses approximate second-order information for weight quantization; AWQ uses activation statistics to protect important weight channels; SmoothQuant rescales to reduce activation-outlier difficulty; QuaRot uses rotations to make quantization easier. [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [SmoothQuant](https://arxiv.org/abs/2211.10438), [QuaRot](https://arxiv.org/abs/2404.00456).

**Derived metadata example:** 4-bit weights grouped in sets of 64 with one FP16 scale per group use 4 + 16/64 = 4.25 bits per weight before any zero points, padding, or other metadata. Raw 4-bit arithmetic is therefore not the same as 4-bit total storage.

KV can need a different format from weights. KIVI finds that keys and values favor different grouping directions, motivating per-channel key and per-token value quantization. Its total memory savings are not simply the nominal change in KV bit width because other tensors and buffering remain. [KIVI](https://arxiv.org/abs/2402.02750).

### 5.2 What ternary weights change

For weights restricted to minus one, zero, or plus one, multiplication by a weight becomes sign selection and optional omission:

\[
y_j=\sum_{i:w_{ij}=1}x_i-\sum_{i:w_{ij}=-1}x_i.
\]

This can eliminate general multipliers from the corresponding linear operation. It still needs accumulation, scales, activation handling, state, and the model's other operators. The term “1-bit LLM” does not imply one-bit activations, one-bit accumulators, or one-bit representations for every tensor.

BitNet b1.58 defines ternary-weight language modeling; the 2B4T report provides an open native ternary model trained on four trillion tokens. These are models trained for the format. Their results do not establish that an arbitrary pretrained dense model can be rounded to ternary with equivalent quality. [BitNet b1.58](https://arxiv.org/abs/2402.17764), [BitNet b1.58 2B4T](https://arxiv.org/abs/2504.12285).

### 5.3 Lookup-based computation

Low-bit weights permit precomputing a table of partial sums for a small activation group. Weight codes select table entries, and the selected entries are accumulated. Bit-plane decomposition can extend the mechanism to several bit widths. This exchanges general multiplication for lookup, precomputation, and accumulation. Table size, replication, ports, and reuse determine whether that exchange pays off.

T-MAC is an open implementation using lookup-based mixed-precision matrix multiplication on CPUs, with an NPU extension also described in its repository. Bitnet.cpp provides ternary-specific lookup and packed-integer paths. Their CPU performance is evidence for the implemented kernels on those systems; an RTL version would need its own timing, resource, and bandwidth measurements. [T-MAC](https://github.com/microsoft/T-MAC), [Bitnet.cpp paper](https://aclanthology.org/2025.acl-long.457/).

TerEffic is a dedicated FPGA architecture with 1.6-bit packed weight storage, a ternary decoder, a TMat core, normalization, and activation/state buffers. It evaluates fully on-chip and HBM-assisted variants using MatMul-free LM demonstration models. Its 370M on-chip example uses two U280 FPGAs, not a single FPGA holding the whole model. Its reported power comes from Vivado power reports. [TerEffic](https://arxiv.org/html/2502.16473v2).

### 5.4 Sparsity has several incompatible meanings

| Kind | What is sparse | Additional hardware work |
| --- | --- | --- |
| Weight sparsity | Stored matrix entries | Index decoding, irregular reads, load balance |
| Structured N:M or block sparsity | A regular subset of entries/blocks | Format-specific selectors and mapping |
| Activation sparsity | Intermediate values or selected neurons | Detection or prediction and selective weight access |
| MoE | Selected expert networks for a token | Routing, token grouping, expert scheduling, communication |
| Attention sparsity | Selected token interactions | Position selection and KV gathers |

Sparse support for one pattern does not imply efficient support for the others. Metadata and movement can cost more than the arithmetic removed. Input-dependent activation pruning can also change the function being computed.

Deja Vu studies predicted contextual sparsity; PowerInfer places frequently active neurons on a consumer GPU with CPU support for less frequent ones; TEAL applies magnitude-based activation sparsity without continued training. Their model coverage and approximation assumptions matter, especially when moving from ReLU-heavy models to SiLU/SwiGLU families. [Deja Vu](https://proceedings.mlr.press/v202/liu23am), [PowerInfer](https://arxiv.org/abs/2312.12456), [TEAL](https://arxiv.org/abs/2408.14690).

## 6. Model families with different hardware requirements

### 6.1 Pure recurrent state-space models

A simplified selective state-space update is

\[
s_t=\bar A_t\odot s_{t-1}+\bar B_t x_t,\qquad y_t=C_t s_t.
\]

The exact dimensions and parameterization depend on the model. For fixed model/state dimensions, a pure recurrent model does not retain a growing Transformer KV cache. Its state and work per decoded token can be independent of context length. Dense projections still consume weights and arithmetic.

Mamba's selective mechanism uses input-dependent state parameters. Mamba-2's state-space duality exposes chunked algorithms that use matrix multiplication more effectively. Consequently, recurrent decode and chunked prefill/training can favor different hardware organizations within the same model family. A design for original Mamba should not be presumed optimal for Mamba-2. [Mamba](https://arxiv.org/abs/2312.00752v2), [Mamba-2](https://arxiv.org/abs/2405.21060).

The hardware workload includes dense projections, state reads/writes, elementwise updates, gates, normalization, and nonlinear functions. Recurrent state precision can affect long-run behavior because errors feed into later updates. The useful state capacity is bounded by the architecture: fixed memory does not imply exact retention of arbitrary amounts of past information.

### 6.2 Gated DeltaNet and matrix-valued state

For column-vector keys/values, a simplified linear-attention state is S_t = S_(t-1) + v_t k_t-transpose, with output S_t q_t. A gated delta update can be written

\[
\widetilde S_t=\alpha_t S_{t-1},\qquad
S_t=\widetilde S_t+\beta_t(v_t-\widetilde S_tk_t)k_t^\top.
\]

This makes the work concrete: decay a matrix, multiply it by a key, form an error vector, and apply an outer-product update. The exact gating, normalization, and chunk algorithm must follow the selected model. The Gated DeltaNet paper combines global forgetting with targeted delta-rule updates and provides a hardware-efficient parallel training algorithm. [Gated Delta Networks](https://arxiv.org/html/2412.06464).

The hardware-relevant resources include state SRAM, broadcast, reductions, outer-product accumulation, and sufficiently fast state updates. This is a different recurring workload from scanning a conventional Transformer KV cache.

### 6.3 RWKV

RWKV uses a recurrent formulation for decoding while retaining parallelizable training formulations. RWKV-7 introduces expressive dynamic state evolution using a generalized delta-rule formulation. Its paper releases models ranging from 0.19B to 2.9B parameters and reports constant memory and per-token inference work with respect to context length. Dense projection weight traffic remains relevant. [RWKV-7](https://arxiv.org/abs/2503.14456).

### 6.4 Hybrid recurrent and attention models are deployed model families

Qwen3-Next-80B-A3B combines Gated DeltaNet, gated full attention, sparse MoE, and multi-token prediction. Its official card specifies 48 layers with a repeated three-linear-attention / one-full-attention pattern, 80B total / 3B active parameters, 512 routed experts with 10 selected, and one shared expert. This is direct evidence that recurrent-state compute and standard attention can coexist in one model. [Qwen official model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

Nemotron 3 Super's April 2026 report describes a 120B total / 12B active hybrid Mamba-attention MoE model, LatentMoE, native speculative-decoding support through MTP, and NVFP4 pretraining. Its throughput comparisons involve particular context lengths, precisions, and software; they do not isolate architecture alone. [Nemotron 3 Super](https://arxiv.org/html/2604.12374v1).

A hybrid model retains both recurrent state and KV for its attention layers. “Contains Mamba” does not mean “has no KV cache.”

### 6.5 Mixture of experts

For a token x, an MoE layer evaluates selected experts and combines their outputs:

\[
y=\sum_{e\in\operatorname{TopK}(g(x))}p_e(x)f_e(x).
\]

Fewer active parameters reduce arithmetic per token, while total parameters still determine weight capacity. Different tokens can select different experts. Larger batches can improve reuse within an expert but also activate a larger fraction of all experts. Efficient execution needs grouping, load balancing, expert placement, and often all-to-all communication.

DeepSeek-V3 has 671B total / 37B active parameters; its MoE layers use eight selected routed experts plus a shared expert. The gpt-oss model card provides another concrete family: gpt-oss-20b has 20.9B total / 3.6B active parameters, 24 layers, and four selected experts from 32 per MoE layer. These are distinct matrix-shape and routing workloads despite both being MoEs. [DeepSeek-V3](https://arxiv.org/html/2412.19437v2), [gpt-oss architecture](https://deploymentsafety.openai.com/gpt-oss/architecture).

Active parameter counts should never be substituted for total model-storage requirements. A model with only a few billion active parameters can still require tens or hundreds of gigabytes of stored expert weights.

**Derived routing illustration:** if each token independently selects k distinct experts uniformly from E experts, the expected number of distinct experts touched by B tokens is E[1-(1-k/E)^B]. As B grows, the batch tends to touch most experts even though each token uses only k. Actual routers are not uniformly independent, but this calculation illustrates why per-token active weights and per-batch weight traffic are different quantities.

The gpt-oss card also specifies alternating windowed/full attention, a 128-token attention window, LayerNorm, and an unconventional SwiGLU with clamping and a residual connection. Operator names alone are insufficient to establish that a generic implementation matches a particular checkpoint. [gpt-oss architecture](https://deploymentsafety.openai.com/gpt-oss/architecture).

### 6.6 Diffusion language models

Masked diffusion models such as LLaDA refine multiple masked token positions through repeated model evaluations. Their parallelism differs from one-token autoregressive decode, but they may do multiple full or block forward passes. Total work depends on denoising steps, block size, cache policy, and selection rules. Parallel token generation alone does not establish lower latency or better quality. [LLaDA](https://arxiv.org/abs/2502.09992).

Bidirectional attention also complicates reuse of old K/V when positions change. Fast-dLLM introduces block-wise approximate cache reuse and confidence-aware parallel decoding. That is different from the exact append-only KV cache in ordinary causal decoding. [Fast-dLLM](https://arxiv.org/html/2505.22618).

The January 2026 paper “Beyond GEMM-Centric NPUs” studies diffusion sampling: vocabulary logits, reduction-based token selection, masked updates, and mixed integer/floating-point buffers. It uses cycle-accurate simulation and post-synthesis RTL verification, not a fabricated diffusion ASIC. This supplies concrete prior art for hardware that accelerates a language-model sampling pipeline rather than only GEMM. [Paper, version 1](https://arxiv.org/html/2601.20706v1).

### 6.7 MatMul-free models and fixed-base adaptation

Scalable MatMul-free Language Modeling combines low-bit linear operations with a recurrent architecture and evaluates models up to 2.7B parameters. The term describes its specific computational design; it should not be read as eliminating every multiply, reduction, nonlinear function, or output operation in an inference system. TerEffic uses models from this family for its demonstrations. [MatMul-free LM](https://arxiv.org/abs/2406.02528), [TerEffic](https://arxiv.org/html/2502.16473v2).

LoRA represents a weight update as a low-rank product. With row-vector activations, y = xW + (xA)B, where A is d_in by r and B is r by d_out. **Derived:** evaluating an unmerged adapter adds about 2r(d_in+d_out) operations per token per adapted linear layer. Base weights can remain fixed while adapters change, although a hardware implementation must provision the adapter storage, arithmetic, and movement. Taalas explicitly claims LoRA support for its largely hardwired Llama implementation. [LoRA](https://arxiv.org/abs/2106.09685), [Taalas disclosure](https://taalas.com/the-path-to-ubiquitous-ai/).

## 7. Serving algorithms change the chip workload

### 7.1 Speculative decoding

A draft model proposes tokens, and a target model verifies several positions in parallel. With the correct acceptance/rejection and correction sampling, the algorithm preserves the target model's output distribution. Simple agreement-only heuristics do not establish that property. Speed depends on acceptance, draft cost, target verification, and available bandwidth/compute. [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html).

Hardware-visible state must match the accepted prefix. A Transformer can discard speculative KV entries beyond that prefix. A recurrent model updates the same state repeatedly, so correct recovery can require snapshots, saved intermediate states, or replay. That state-management requirement is a consequence of the update semantics; it is not evidence for a particular new accelerator design.

**Derived acceptance illustration:** for a draft window of gamma tokens, a constant independent conditional acceptance probability alpha, and the standard correction/bonus token, expected advancement is 1 + alpha + ... + alpha^gamma. This is not the speedup: drafting, verification, state recovery, and communication take time, and acceptance is workload-dependent.

### 7.2 Different kinds of disaggregation

| Split | What crosses the boundary | Cost that can defeat the split |
| --- | --- | --- |
| Prefill versus decode | Built KV state, request handoff | Cache transfer, queueing, imbalance |
| Attention versus MLP/experts | Hidden vectors and routing metadata per layer | Repeated communication startup and synchronization |
| Draft versus target | Proposals, probabilities or validation data, state coordination | Draft overhead and rejected work |
| Tensor/expert parallel devices | Activations, reductions, routed tokens | Collectives, network topology, stragglers |

DistServe separates prefill and decode to reduce interference and choose phase-specific resource allocation. The benefit depends on its TTFT/TPOT targets and available transfer bandwidth. [DistServe](https://arxiv.org/html/2401.09670v3).

An October 12, 2026 d-Matrix disclosure describes a different split within the model: an H100 manages attention, KV, embeddings, and routing while Corsair processes experts and shards the output head. In its gpt-oss-20b example, expert weights at 4-bit precision fit into an eight-card Corsair server with 16 GB total SRAM; the pipeline communicates over JetStream. The disclosure reports 24 layer round trips plus an output-head round trip. These are vendor-reported implementation facts, not an independent benchmark reproduction. [d-Matrix AFD technical article](https://www.d-matrix.ai/accelerating-ai-inference-with-attention-ffn-disaggregation/).

### 7.3 Flash storage and limited-memory execution

Apple's “LLM in a flash” studies storing weights in flash and bringing needed data to DRAM. It combines activation sparsity, reuse of recently active neurons, and row-column bundling to make reads larger and more contiguous. Its large relative speedups are against naive loading baselines under its tested workloads, not against arbitrary modern GPU serving systems. It illustrates that storage controllers and access layout can matter as much as multipliers for a capacity-constrained deployment. [Apple paper](https://aclanthology.org/2024.acl-long.678/).

## 8. Published hardware architectures and evidence boundaries

The following are established architecture classes and concrete references. “RTL-mappable” means the digital mechanism can be represented in RTL; it does not certify that a full-scale product can be built with the current project's resources.

| Architecture class | Actual hardware organization | Concrete evidence | Boundary of the evidence |
| --- | --- | --- | --- |
| Programmable systolic accelerator | Matrix array, banked scratchpad, DMA, accumulators | [Gemmini](https://github.com/ucb-bar/gemmini) | Open generator and software; complete LLM execution still needs operators and orchestration |
| Sparse/mixed-precision FPGA LLM engine | Configurable sparse DSP chains, local activation management, length-adaptive compilation | [FlightLLM paper](https://arxiv.org/html/2401.03868), [artifact](https://github.com/FlightLLM/flightllm_test_demo) | Artifact distinguishes measured U280 testing from simulated VHK158 performance; demo is not a verified complete reusable RTL package |
| Ternary FPGA inference engine | Packed ternary storage, decoding, add/lookup datapath, normalization and state buffers | [TerEffic](https://arxiv.org/html/2502.16473v2) | FPGA place-and-route/evaluation; power from Vivado estimates; model family and card count matter |
| Reconfigurable Mamba accelerator | PEs switch between reductions and elementwise work; reused nonlinear units | [MARCA](https://arxiv.org/html/2409.11440) | Verilog synthesis in a TSMC 28 nm library, cycle simulation, estimated memory/power; not fabricated silicon |
| Attention processing in memory | PIM units near HBM banks/bank groups or buffer die; xPU does other work | [AttAcc simulator](https://github.com/scale-snu/attacc_simulator/) | Modified Ramulator-based architecture simulation, not an off-the-shelf RTL HBM implementation |
| NPU plus PIM | NPU does compute-heavy GEMM; PIM serves attention GEMV; scheduling overlaps work | [NeuPIMs](https://github.com/casys-kaist/NeuPIMs) | Public cycle-accurate simulator; proposed memory changes require suitable physical memory technology |
| Diffusion sampling NPU | Vector/reduction instructions, masked updates, SRAM reuse, separated precision domains | [Diffusion NPU v1](https://arxiv.org/html/2601.20706v1) | Simulation and post-synthesis verification; no fabricated ASIC established |
| Model-specific fixed silicon | Model-specific weight/computation realization plus limited configurability | [Taalas HC1](https://taalas.com/the-path-to-ubiquitous-ai/) | Company reports silicon and demo/API; performance and quality are tied to its quantized model |
| Deterministic spatial streaming | Compile-time schedules for compute, SRAM, and interchip transfers | [Groq Hot Chips](https://hc34.hotchips.org/assets/program/conference/day2/Machine%20Learning/HotChips34%20-%20Groq%20-%20Abts%20-%20final.pdf), [LPX update](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) | Older TSP and newer LPX are different generations; do not mix their specs |
| Heterogeneous GPU and SRAM accelerator | Assign different phases or sublayers to different devices | [d-Matrix AFD](https://www.d-matrix.ai/accelerating-ai-inference-with-attention-ffn-disaggregation/), [NVIDIA LPX](https://www.nvidia.com/en-us/data-center/lpx/) | Vendor implementations/claims; whole-system cost includes every device and link |
| Analog compute in memory | Conductance-coded weights and analog dot products, plus conversion and digital processing | [IBM 64-core chip](https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference) | Fabricated 14 nm chip, demonstrated ResNet/LSTM; not evidence of a complete competitive LLM system |
| Photonic interconnect | Optical links and packaging between compute/memory devices | [Lightmatter Passage](https://lightmatter.com/products/passage/) | Interconnect product; optical IO does not by itself implement attention or matrix arithmetic |

### 8.1 What processing in memory changes

PIM reduces traffic across the external memory interface by doing some operations closer to stored data. It does not remove internal reads, row constraints, bank conflicts, accumulation, data distribution, or precision limitations. NeuPIMs explicitly addresses overlap between memory and PIM operations with dual-row-buffer organization and scheduling. AttAcc explores PIM placement at multiple HBM hierarchy levels. [NeuPIMs](https://github.com/casys-kaist/NeuPIMs), [AttAcc](https://github.com/scale-snu/attacc_simulator/).

An RTL simulation can model a proposed PIM device, but a fabricated implementation requires a memory technology, macros, or partner capable of that organization. Ordinary external HBM cannot be turned into bank-level PIM by adding compute RTL to a separate chip.

### 8.2 Analog computation and optical transport solve different physical problems

In resistive analog compute-in-memory, conductances represent weights and electrical behavior forms dot products. Peripheral conversion, precision, noise, programming, and digital operators remain important. IBM's published 64-core chip includes digital activation processing and interconnect alongside analog cores. Its reported workload coverage is ResNet and LSTM; extending the result to a full Transformer requires additional evidence. [IBM primary publication](https://research.ibm.com/publications/a-64-core-mixed-signal-in-memory-compute-chip-based-on-phase-change-memory-for-deep-neural-network-inference).

Lightmatter Passage is a photonic interconnect platform. Its role is moving data between devices using optical connectivity and packaging. Any claim about a complete LLM speedup needs the surrounding compute/memory system and workload. It should not be described as a chip that runs an entire model's matrix multiplies optically merely because it uses photonics. [Passage](https://lightmatter.com/products/passage/).

## 9. What companies publicly disclose as of the research date

### 9.1 Current architecture directions and status

| Company/platform | Disclosed architecture focus | Evidence and dated status |
| --- | --- | --- |
| NVIDIA Blackwell / Vera Rubin | Tensor engines, low precision, large HBM systems, NVLink/network integration | Blackwell technical brief and Rubin January 2026 announcement establish architecture; a specific October 21 Google A5X announcement says its GA is coming in November, despite the headline's wording. [Blackwell brief](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief), [Rubin announcement](https://nvidianews.nvidia.com/news/rubin-platform-ai-supercomputer), [A5X timing](https://blogs.nvidia.com/blog/google-cloud-vera-rubin-infrastructure/) |
| NVIDIA Groq 3 LPX | SRAM-based deterministic execution, low-latency interchip schedules; pairs with Rubin | NVIDIA announced full production August 24, 2026. Architecture discussion specifies 256 LPUs and 128 GB total SRAM per LPX rack. [Production announcement](https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Groq-3-LPX-Now-in-Full-Production-With-World-Class-Speed-for-Agentic-AI/default.aspx), [technical article](https://developer.nvidia.com/blog/how-nvidia-groq-3-lpx-unlocks-ultrafast-interactivity-at-long-context-on-nvidia-vera-rubin/) |
| Groq | Deterministic compiler-scheduled tensor streaming and SRAM placement | Groq's December 24, 2025 statement describes a nonexclusive inference-technology license to NVIDIA. This does not make every legacy Groq product identical to LPX. [Groq announcement](https://groq.com/newsroom/groq-and-nvidia-enter-non-exclusive-inference-technology-licensing-agreement-to-accelerate-ai-inference-at-global-scale) |
| Google TPU | Ironwood plus separate eighth-generation training and serving designs | April 22, 2026 deep dive describes TPU 8t and 8i; the inspected product page lists both as “Coming soon” and Ironwood as generally available. [8t/8i](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive), [availability page](https://cloud.google.com/tpu) |
| AMD Instinct | Chiplets, HBM, matrix/vector engines, low precision, rack-scale fabrics | MI455X product page specifies CDNA5, July 23, 2026 launch, 432 GB HBM4, 23.3 TB/s peak bandwidth. A product launch/specification does not prove every partner configuration is deployed. [MI455X](https://www.amd.com/en/products/accelerators/instinct/mi400/mi455x.html) |
| AWS Trainium3 | Tensor/vector/scalar/GPSIMD engines; explicit local memory, DMA, collectives | Official architecture documentation describes eight NeuronCore-v4 cores, 144 GiB HBM, 4.9 TB/s bandwidth, and 256 MiB SBUF. [Trainium3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html) |
| Microsoft Maia 200 | FP4/FP8 tensor cores, SRAM/DMA/NoC, Ethernet-based scale-up | January 26, 2026 announcement reports deployment in US Central and gives 216 GB HBM3e, 7 TB/s, 272 MB SRAM. [Microsoft announcement](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) |
| Cerebras WSE-3 | Wafer-scale cores with distributed on-wafer SRAM and fabric | Hot Chips 2024 gives 900,000 cores, 44 GB on-chip memory, 21 PB/s aggregate memory bandwidth. Inference access was announced in August 2024. Aggregate on-wafer bandwidth is not directly comparable to off-chip HBM bandwidth. [Hot Chips](https://hc2024.hotchips.org/assets/program/conference/day2/72_HC2024.Cerebras.Sean.v03.final.pdf), [inference launch](https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed) |
| d-Matrix Corsair | Memory-centric chiplet system with SRAM, DRAM, and GPU interoperability | October 12, 2026 disclosure describes AFD with H100; Squadron announcement says Corsair servers/racks generally available and larger clusters shipping to priority customers. [AFD](https://www.d-matrix.ai/accelerating-ai-inference-with-attention-ffn-disaggregation/), [Squadron](https://www.d-matrix.ai/announcements/squadron-compute-clusters/) |
| SambaNova SN50 | Reconfigurable dataflow and three-tier memory: SRAM, HBM, large-capacity memory | February 24, 2026 announcement says shipping planned for second half of 2026. That announcement alone does not verify a completed customer deployment. [SN50](https://sambanova.ai/blog/introducing-the-sn50-rdu-purpose-built-for-agentic-inference) |
| Tenstorrent Blackhole | Grid of Tensix compute cores, local SRAM, dataflow processors, NoC, tiled execution | Public hardware documentation and open TT-Metal software. Specified processor has 120 Tensix cores and 180 MB SRAM; individual cards can expose different enabled resources. [hardware docs](https://docs.tenstorrent.com/aibs/blackhole/), [programming guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md) |
| Taalas HC1 | Model-specific silicon combining fixed model realization and storage/computation specialization | Company describes a Llama 3.1 8B demonstrator and beta API. It reports mixed 3-/6-bit parameters, some quality degradation, configurable context, and LoRA support. [disclosure](https://taalas.com/the-path-to-ubiquitous-ai/), [product](https://taalas.com/products/) |
| Etched | Current disclosure emphasizes low-voltage inference, splittable math arrays, HBM/SRAM hybrid cluster memory | June 30, 2026 update says A0 silicon returned and rack-scale product under customer validation. This is more current than describing the company only through its original transformer-only Sohu pitch. [Etched update](https://www.etched.com/progress/frontier-inference-clusters) |
| MatX One | Splittable systolic array, SRAM-first latency plus HBM context capacity | February 24, 2026 announcement says development/manufacturing work and tapeout in under a year. It is an announced design, not demonstrated shipping performance in that source. [MatX](https://matx.com/research/series_b) |
| Positron Asimov | Commodity LPDDR5X, reconfigurable systolic geometry, vector operators, on-chip CPUs | Inspected product page labels it coming in 2027. Its bandwidth/economic claims are roadmap/vendor claims, not a current production benchmark. [Asimov](https://www.positron.ai/asimov) |
| Lightmatter | Photonic links, optical IO, interconnect packaging | Passage documentation supports a transport/interconnect role; complete model execution needs external compute and memory. [Passage](https://lightmatter.com/products/passage/) |

### 9.2 Google's split exposes specific workload differences

The TPU 8t/8i technical deep dive gives the following announced specifications:

| Disclosed feature | TPU 8t | TPU 8i |
| --- | --- | --- |
| Primary emphasis | Large-scale pretraining | Sampling, serving, reasoning |
| Scale-up topology | 3D torus | Boardfly |
| Specialized block | SparseCore and decoder-related support | Collectives Acceleration Engine |
| HBM capacity | 216 GB | 288 GB |
| Vector SRAM | 128 MB | 384 MB |
| HBM bandwidth | 6,528 GB/s | 8,601 GB/s |

The 8i design explicitly spends resources on SRAM and reductions/synchronization. These are vendor-published specifications and architectural explanations. They do not establish that every inference workload prefers 8i. [Google technical deep dive](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive).

### 9.3 A particularly model-specific implementation

Taalas reports 17,000 output tokens/s per user for its HC1 Llama 3.1 8B demonstrator, with a stated 1K/1K sequence setting. It also discloses aggressive mixed 3-/6-bit quantization and some quality degradation. The product page gives an 815 mm², 53B-transistor, TSMC 6 nm chip and a 2.5 kW server. The performance number, quantization, quality, model identity, and server boundary belong together; dropping those conditions would make a comparison misleading. [Taalas architecture/quality disclosure](https://taalas.com/the-path-to-ubiquitous-ai/), [HC1 product](https://taalas.com/products/).

“Weights loaded into an RTL ROM in Verilator” is not equivalent evidence to such a fabricated implementation. RTL constants can infer different structures depending on synthesis and memory macros. Simulation alone does not establish area, frequency, physical memory bandwidth, power, or manufacturing feasibility.

## 10. What can be investigated at different implementation levels

| Mechanism | What can be represented in a small digital prototype | What remains a separate scaling/physical question |
| --- | --- | --- |
| Dense GEMM/GEMV | MACs, tiling, accumulators, local buffers | Large-model capacity, SRAM ports, wire timing, memory interfaces |
| Attention tiling | Online softmax, cache streaming, causal masking | Long-context bandwidth and large state placement |
| Ternary/LUT arithmetic | Packed weight decoding, lookup/addition, scale handling | Large-table distribution, routing, accuracy and model support |
| Recurrent update | Fixed-size state, gates, outer products, reductions | Long-run numerical stability and projection bandwidth |
| MoE routing | Small router, top-k, expert dispatch | Many-device all-to-all, imbalance and total capacity |
| Speculative execution | Accepted-prefix accounting, snapshots/replay | Useful speedup under real acceptance and traffic patterns |
| Diffusion sampling | Vector reductions, top-k/masked updates, buffer reuse | Full model/denoising performance and cache correctness |
| PIM | Digital functional/cycle model of near-memory compute | Actual compatible DRAM/HBM device implementation |
| Analog/photonic hardware | Digital controllers and behavioral models | Device models, mixed-signal circuits, optical/analog fabrication |

This table is an engineering inference from the digital operations and cited implementation examples, not a ranking or selection of the project's architecture.

Open resources establish different starting points. Gemmini is a hardware generator; SCALE-Sim models systolic-array performance; AttAcc and NeuPIMs are architecture simulators; T-MAC is a software kernel implementation; TT-Metal is software for existing hardware. None should be described interchangeably as a complete open-source LLM ASIC. [Gemmini](https://github.com/ucb-bar/gemmini), [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim), [AttAcc](https://github.com/scale-snu/attacc_simulator/), [NeuPIMs](https://github.com/casys-kaist/NeuPIMs), [T-MAC](https://github.com/microsoft/T-MAC), [TT-Metal](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md).

## 11. Facts needed to interpret performance and originality

### 11.1 Comparable performance needs a complete workload

| Field | Why it matters |
| --- | --- |
| Exact checkpoint and architecture | Equal parameter counts do not imply equal computation or quality |
| Weight, activation, accumulation, state, and KV precision | Different resources and accuracy tradeoffs |
| Batch/concurrency and scheduling | Aggregate throughput can rise while user latency worsens |
| Input/output/context lengths | Prefill and cache costs change substantially |
| Time to first token and time per output token | Distinct user-visible phases |
| Accepted output tokens | Speculative proposals and diffusion iterations are not all final output |
| Number of chips/cards/racks | A per-user rate does not reveal system cost |
| Measured versus modeled clock and memory | Simulation cycles alone do not determine seconds |
| Power boundary | Chip estimate, board measurement, server, and facility power differ |
| Accuracy and task coverage | Quantization, approximation, pruning, and decoding can alter quality |
| Baseline implementation and date | A weak or obsolete kernel can inflate a relative speedup |

FLOPS or TOPS alone does not determine any of those end-to-end rates. Peak sparsity figures must specify their supported sparsity format; FP4, FP8, BF16, integer, dense, and sparse peak rates are not interchangeable. The serving analyses and the hardware papers above repeatedly expose these distinctions. [Inference analysis](https://arxiv.org/abs/2211.05102), [DistServe](https://arxiv.org/html/2401.09670v3), [FlashAttention-4](https://arxiv.org/html/2603.05451).

### 11.2 Established prior art already covers broad labels

| Broad label | Existing concrete precedent |
| --- | --- |
| “LLM systolic accelerator” | TPU-style matrix processing, Gemmini-based exploration, FPGA LLM engines |
| “Ternary LLM accelerator” | Bitnet.cpp/T-MAC software mechanisms and TerEffic FPGA architecture |
| “Mamba accelerator” | MARCA |
| “PIM for LLM attention” | AttAcc and NeuPIMs |
| “Diffusion LLM hardware” | January 2026 diffusion sampling NPU work |
| “An entire model specialized into silicon” | Taalas HC1 disclosure |
| “GPU plus specialized decode hardware” | LPX and heterogeneous serving implementations |
| “Attention and experts on different chips” | d-Matrix's October 2026 AFD disclosure |
| “Flexible array for GEMM and GEMV” | MatX/Positron disclosures and broader array-design literature |

These precedents establish that the broad categories are occupied. They do not establish that every narrower contribution is exhausted. A novelty claim would require a defined mechanism, workload, comparison, and a focused literature/patent review. This survey collects the evidence needed to think about those distinctions without choosing a design.

### 11.3 Relationship to the existing project

The local [PLAN.md](../PLAN.md) describes a two-layer GPT-2-style model with d_model = 64, two heads, 256-token vocabulary, context up to 32, INT8 weights/activations, and INT32 accumulation, running prefill and decode in Verilator. Its status paragraph is dated July 5, 2026; that paragraph is a historical project note, not a fresh verification of tests or implementation completeness.

**Derived from that stated configuration:** ignoring biases and norms and assuming tied token embeddings, its main matrices and embeddings total about 116,736 parameters. At INT8, their raw storage is roughly 114 KiB. This is a useful correctness-scale workload, but its weight/cache capacity and reuse regime are very different from a multi-billion-parameter deployment. Measurements on it need their actual scale stated.

The local plan's all-on-chip simulation environment can establish functional behavior and cycle counts for the modeled design. Physical area, achievable clock, SRAM realization, external bandwidth, board power, and large-model performance require further evidence. No implementation changes or architecture commitment are part of this research reference.
