# Nonlinear units and MoE

Your array only multiplies and adds. A transformer also needs GELU, softmax and LayerNorm, and newer models add experts. This chapter works out how each of those fits on your chip: the 1×16 array with four multipliers per PE from *Systolic arrays*, at context 32.

The short version:

- Because your data is int8, a 256-entry lookup table gives you GELU with no approximation work at all. [D]
- All of these functions fit in about 1 KB of tables. [D]
- Run one step at a time, the work around the array takes 4,160 element steps against 1,920 array beats. Fused into the array's output path with 4 lanes, it adds only about 8%. [D]
- MoE is much easier on your chip than in a datacenter, because almost all of its problems come from having many users and many chips.
- MoE saves compute, not storage. On a chip where weights live on the die, that is the whole decision.

These numbers come from `model/nonlinear.js` and `tokenPlan` in `model/model.js`. They are beat counts and integer experiments on synthetic data. None of it is synthesis, and no trained model was involved.

## What a lookup table is

A lookup table is a list of answers worked out in advance. The input is the address, and the output is whatever is stored there.

A tiny example: say your input can only be 0, 1, 2 or 3, and you want its square. Store 0, 1, 4, 9. Input 3 reads slot 3 and gives 9. No multiplier ran. The maths happened once, on your laptop, before the chip existed.

That works for any function, however ugly. GELU has a tanh and a cube inside it, and the hardware doesn't care. It reads a slot.

One naming warning. On an FPGA, "LUT" also means the small 6-input logic cell everything is built from. That is a different thing. In this chapter a LUT is a table that holds a function.

## Why int8 makes GELU exact

The size of a table is 2 to the power of the input bits. So input width decides everything.

Your FFN output is requantized to int8 before GELU. That means only 256 inputs can ever happen. List all 256 answers and you have covered every case the chip will ever see.

A walk-through, with values scaled so int8 covers about −6 to 6, which is 0.047 per step: [D]

- Input −20 means −0.945. GELU gives −0.163. Stored as −3.
- Input 0 means 0. Stored as 0.
- Input 20 means 0.945. GELU gives 0.782. Stored as 17.
- Input 60 means 2.835. GELU gives 2.83. Stored as 60.

The worst error over all 256 entries is 0.498 of one output step. [D] That is just rounding to int8, which you were doing anyway. So the table is as good as a perfect GELU unit followed by rounding, and it costs 256 bytes.

The scale matters, though. The table is built for one input scale and one output scale. If your two layers calibrate to different scales, you need two tables, which is still only 512 bytes.

## When the input is too wide

Now suppose the input were 16 bits. A direct table needs 65,536 entries. At 2 bytes each that is 128 KB, which is bigger than all your weights at 115 KB. [D] That is the cliff: 8 bits costs 256 bytes, and 16 bits costs more than the model.

The fix is to store a few points and draw straight lines between them. This is the interpolated table: look up two neighbouring points, then do one multiply-add to land between them.

For GELU across −6 to 6, measured against your int8 step size: [D]

- 16 segments: 34 bytes, worst error 1.0 steps. Not good enough.
- 32 segments: 66 bytes, worst error 0.29 steps. This is the sweet spot.
- 64 segments: 130 bytes, worst error 0.07 steps.

So you never need a 128 KB table. But you also don't need interpolation yet, because your input is int8. It becomes worth having if you later feed GELU from wider values to avoid rounding twice.

The third option is a polynomial, which reuses multipliers instead of storing anything. I-BERT does GELU and softmax this way with integers only. [O] It suits a chip that already has spare vector multipliers. You don't, and tables are smaller at this size.

## Softmax, step by step

Softmax is the awkward one, because it needs a whole row before it can finish. Take four scores, in int8 with 0.063 per step: 30, 20, 10, −40.

**Step 1: find the max.** It is 30. Run a comparator over the scores as they leave the array, and this costs nothing extra.

**Step 2: subtract.** You get distances 0, 10, 20, 70. These can't be negative, and they are small whole numbers. So exp has become a lookup with distance as the address.

**Step 3: look up.** The table holds 127 × exp(−distance × 0.063), which gives 127, 68, 36, 2. Only the first 88 entries are non-zero, so past that the answer is just 0. [D]

**Step 4: add them up.** The sum is 233.

**Step 5: multiply by V.** With V values 100, −50, 20, 8, the array computes 127×100 + 68×(−50) + 36×20 + 2×8 = 10,036.

**Step 6: divide once.** 10,036 ÷ 233 = 43.07. Doing it in floating point gives 43.28. [D]

Look at the order in steps 5 and 6. You would expect to divide first, turning each value into a probability, then multiply by V. Don't.

Your array takes signed int8, so its inputs have to fit in 0 to 127. Imagine 32 equal scores. Each probability is 1/32, which scales to about 4. Your 8-bit input now holds a value with only two useful bits. If you divide last instead, all 32 inputs stay at 127 and nothing is lost.

Measured attention output error, over 20,000 random rows, in output steps: [D]

- Flat scores: 0.41 when dividing first, 0.03 when dividing last.
- Medium scores: 0.42 first, 0.17 last.
- Peaky scores: 0.28 first, 0.12 last.

The worst single cases are closer together, about 2 to 3 steps either way on the medium and peaky rows. So this helps the average a lot and flat rows the most. And it is also less work: 32 divides become 1, and that one can ride along in the requant scale, so it needs no extra multiplier.

The trick you may have read about, computing exp as a power of 2, splits the job into a shift and a small table. It is the right tool for wide inputs, and Softermax builds on it. [O] On your int8 scores a direct table is better: a plain 16-entry version is off by 2.7 out of 127 with no interpolation, and your 88 entries are just rounding. [D]

## Reciprocal and inverse square root

Softmax needs 1/sum. LayerNorm needs 1/√variance. Both take wide inputs, so a direct table won't do. The method for both has two parts:

- **Shift** the number into a small range, 1 to 2 for reciprocal and 1 to 4 for square root. This is just counting leading zeros. The 4 is there so the shift stays even, because the square root halves it.
- **Seed and refine.** Read a rough answer from a small table, then run one Newton step, which roughly doubles the correct digits.

With 32 entries: [D]

- Reciprocal: 1.5% error from the table, 0.024% after one Newton step.
- Inverse square root: 2.3% from the table, 0.076% after one Newton step.

Int8 output needs about 0.2%, so one step is enough. Using 16 entries gives 0.28% for square root, which is too close. The Newton step is 2 or 3 multiplies, so one shared multiplier handles it in a few beats. I assumed 8 beats for the whole thing.

## LayerNorm without the pain

Three facts make LayerNorm cheap on your chip.

**Statistics come free.** As the residual values stream past, keep a running sum and a running sum of squares. In integers, 64·Σx² − (Σx)² is exact, which I checked against the two-pass version. [D] Dividing by 64 is a shift, because d_model is a power of 2. The sum of squares needs 21 bits. [D]

**Gamma and beta can disappear.** They are constants, so fold them into the next weight matrix and its bias when you export weights. Your chip then only does (x − mean) × r. Do this in the Python export, so RTL and golden match.

**The matmul doesn't have to wait.** This is the best trick here. Normally you finish the statistics, normalize, then start the matmul. But the maths rearranges:

`W(x − μ)·r  =  r·(Wx − μ·colsum(W))`

So feed raw x straight into the array, and fix the result at the output. The column sums are constants you store. r and μ only have to be ready by the time outputs appear, so the 1/√x latency hides behind the matmul. It costs one small table of column sums and one more multiply-add at the edge. I checked the identity numerically. [D] It takes about 40 beats off a token. [D]

## One block, not five

The naive design has a GELU unit, a softmax unit, a LayerNorm unit and a requant unit. Don't build that.

Your model runs these one at a time, never together, so one pipeline at the array's output can do all of them. Each output value passes through once:

- Scale, round and clip to int8. That is requant.
- Read the table, with the function chosen by the top address bits.
- Add the residual.
- Update the running max, sum and sum of squares.
- Write to the buffer.

The tables together: GELU 256 entries, exp 256, seeds 64. That is about 1 KB with 2-byte entries. [D] A hardwired chip can store them as constants, which synthesize to logic with no write port.

How many lanes? Count the rate. A pass of K = 64 takes 16 beats and gives 16 outputs, so one value per beat. A pass of K = 32 takes 8 beats and gives 16 outputs, so two per beat. So **attention sets your minimum at 2 lanes**, and each lane reading a table needs its own read port or its own copy. This is the readout rule from *Systolic arrays* showing up again: short K puts the pressure on whatever sits after the array.

## What this does to your token time

The array needs 1,920 beats per token. The work around it is 4,160 element steps: 1,664 requants, 960 LayerNorm, 512 softmax, 512 GELU, 256 residual adds and 256 for argmax. [D]

Run those one after another, with 4 lanes, and a token takes 3,032 beats. With 1 lane it takes 6,152. The array sits idle most of the time, which is exactly what this guide is named against.

Fused into the output path: [D]

- 1 lane: 2,649 beats, 38% on top of the array.
- 2 lanes: 2,225 beats, 16%.
- 4 lanes: 2,067 beats, 8%.
- 16 lanes: 2,016 beats, 5%.
- 4 lanes plus the deferred LayerNorm: 2,027 beats, 6%.

Lanes stop paying at 4. Why does the time stop at about 5%? Because a batch-1 token is a chain. Softmax can't start until the scores end, and the next layer can't start until this one ends. Only waiting that overlaps something can hide, and with one token there is nothing else to overlap. A big chip hides this behind other users. You can't.

So: **4 lanes, one shared table block.** 2 lanes with deferred LayerNorm is close, at 2,105 beats. Without deferring, 2 lanes also starves the array, because it eats 4 normalized values per beat.

These figures assume 3 beats through the pipeline, 8 beats for 1/√x and one output buffer. Change them in `tokenPlan` when you have real RTL.

::viz token

## MoE in one example

A dense FFN sends every token through the same big block. MoE keeps several smaller blocks, the experts, plus a tiny router that picks which ones each token uses.

Say you have 4 experts and top-1 routing. A token arrives, and the router scores it: 3, −1, 7, 2. Expert 2 wins. That token runs expert 2 only, and the other three do nothing for this token. The next token might pick a different one.

So you store 4 experts, but only pay to run 1.

## Why batch 1 is the easy case

*Scale-out* lists three MoE problems. Look at what happens to each on your chip:

- **Uneven load** means some experts get swamped while others idle. You have one token at a time, so there is no load to balance.
- **Not enough rows** means each expert gets too few tokens to fill the array. You are already at M = 1, so you lose nothing.
- **All-to-all traffic** means tokens crossing between chips. You have one chip.

All three are about many users across many chips. They simply don't exist here.

What stays is the one thing MoE can't avoid: **storage grows with experts stored, and compute grows with experts run.** In a datacenter, or on a laptop with DRAM, storage is the cheap side. Llama 4 Maverick reads about 17 GB of its 400 GB per token at one byte per weight, so at 100 GB/s it would run about 6 tokens a second where a dense 400B runs one every 4 seconds. [R, D] But your weights sit on the die, so storage is area, and area is your most expensive thing.

## What changes in the hardware

Less than you would expect. Five small changes:

**1. The router is one more matmul.** N = 4, K = 64, using 256 bytes of weights per layer. A trap: it should take 4 beats by MAC count, but it takes 16. A pass takes K ÷ Tk beats whether you use 4 columns or 16. [D]

**2. Top-k is a few comparators.** Top-1 of 4 is three compares. Define ties, for example lowest index wins, and use the same rule in Python.

**3. The gate is cheap.** Top-1 needs one scale. For top-2, Mixtral-style softmax over just the two winners equals sigmoid(a − b), so it is one table lookup, and the other gate is 1 minus that. [O]

**4. The expert number becomes address bits.** Lay out the weight ROM so `expert_id` is the top bits. Same width, same reads, same counters.

**5. Apply the gate at the edge.** Scale the expert output in requant, then add it to the residual. Top-2 adds twice.

Now the important part. **Your schedule stays static.** Every token takes the same number of beats, because k is fixed and every expert has the same size. The data only changes *which address* you read, not *how long anything takes*. Your hardcoded FSM gains one register. So the dynamic routing that *Crossbars, NoC* says MoE needs on big machines doesn't apply to you.

One new stall: nothing can start until the router answers. The router adds 16 array beats per layer, and it sits on the critical path. A shared expert, the DeepSeek idea of one expert every token always uses, could run while the router works. [O]

## The three versions you could build

All at 4 lanes, 2,067 beats and 114,688 bytes for the dense model: [D]

- **More knowledge, same speed.** 4 experts, top-1, each width 256. 2,109 beats, 2% slower, but 311,808 bytes, or 2.7× the ROM.
- **Same ROM, much faster.** 4 experts, top-1, each width 64. 1,341 beats, 35% faster, and 115,200 bytes.
- **The middle.** 4 experts, top-2, each width 64. 1,625 beats, 21% faster, 115,200 bytes.

The second one looks like a free win, and that needs care. Each token now sees an FFN a quarter the size. Whether a 2-layer model survives that is a question about model quality, and I have no data on it. Your beats are measured. Your accuracy isn't. Train both in your Python golden and compare before touching RTL.

## Two traps

**Prefill on an array with rows.** On your 1×16, tokens go through one at a time, so routing costs nothing. On an 8×8, it does. 32 tokens spread over 4 experts rarely split evenly, and each expert rounds up to whole tiles. With random routing that averages 5.63 row tiles where dense needs 4, so 41% more, and it can reach 7. [D] It also makes timing depend on the data, which breaks your static schedule. This is another point for the single row.

**Matching your golden model.** In a dense model, a 1-step error stays a 1-step error. In MoE it can pick a different expert, and then the whole FFN output changes. With 4 router scores of typical size 20 steps, the top two are within 1 step on 7.6% of tokens. [D] So the router path must be bit exact with Python, including ties. "Close enough" is not a passing test here.

## Should you add it?

Not first. Your plan already calls numerical matching a key risk, and MoE raises the stakes on it. Get the dense model running end to end, then try MoE as an experiment.

Two things are cheap to do now and keep that door open:

- Leave spare top address bits on the weight ROM.
- Build the output path so a scale can change per pass. Deferred softmax needs that anyway.

## The other things you could add

Each of these is a real option in modern models. Here is what each costs on your chip.

**RMSNorm instead of LayerNorm.** Drop the mean. No running sum, and deferred LayerNorm no longer needs column sums. It is simpler, and you lose nothing in the schedule. If you are free to choose the model, take this one.

**One shared K and V head.** Going from two K and V heads to one takes QKV from N = 192 to 128. Array beats fall from 1,920 to 1,792, the token from 2,067 to 1,939, and the KV cache from 8 KB to 4 KB. [D] It is the cheapest speedup here, and quality needs testing.

**RoPE instead of learned positions.** Replaces a 2 KB position table with 1 KB of sin and cos. [D] Each pair of values needs 4 multiplies, for Q and K, before K is written to the cache. Reorder weight columns at export so each pair leaves the array in the same pass. At context 32 you gain nothing from it, so skip it unless you want longer context.

**SwiGLU instead of GELU.** Three matrices instead of two, with width cut to 176 to stay near the same size. FFN cost goes from 512 to 528 beats per layer. [D] It adds a SiLU table, which is another 256 bytes, and one multiplier. Compute gate and up columns in alternate passes, so you only buffer 16 values.

**Sampling instead of argmax.** Argmax is one comparator on the logits as they stream out. Real sampling normally needs softmax over all 256 logits, which is 256 more exps. The Gumbel trick avoids that: add random noise of a particular shape to each logit, then take argmax as before. This samples exactly from the softmax. I checked it over a million draws, and the frequencies matched to 3 decimal places. [D] You add one LFSR and one table.

**Int4 weights.** ROM falls to 57 KB, and it needs only 32 bytes per beat instead of 64. [D] That is probably your biggest area win if quality holds, since the ROM is likely your largest block, but synthesis has to confirm that.

**Speculative decoding: skip it.** It works by checking 4 guessed tokens in one pass, which is free when a chip is waiting on memory and has compute to spare. Your chip reads weights from on-die ROM at exactly the rate the MACs use them. So checking 4 tokens takes 4× the beats, and even if all 4 are right you only break even. [D] It would save energy on weight reads, but it won't save time.

## Order of work

Each step here keeps your Python golden and RTL matching before you move on.

1. Requant at the array edge, with the rounding rule written down.
2. GELU table, generated by the same script as the golden.
3. Exp table, max and sum, with the divide last.
4. Reciprocal and 1/√x, then LayerNorm with gamma and beta folded in.
5. Measure real beats per token, and compare with 2,067.
6. Try deferred LayerNorm and one K and V head.
7. Train an MoE version in Python, and only build it if quality holds.
