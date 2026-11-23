# 08. Post-design levers: node, die, manufacturing and cost

*Started 2026-11-10. Status: thinking, nothing decided. Numbers come from `die_costs()` and `our_card()` in `calc/qwen35_budget.py`. Wafer prices are analyst estimates, because foundries don't publish them.*

## Two kinds of levers

`02_workload.md` lists four **design levers:** memory bandwidth per dollar, bandwidth utilization, small silicon, and no over-provisioning. Those are decided by the architecture.

**Post-design levers** are about how the chip gets made and sold:
- which foundry and process node
- how big the die is, and its yield
- packaging, testing and the board
- one-time costs (NRE) and how many units share them

They can change the cost a lot without touching the RTL. This note goes through each one, with numbers.

## How a die's cost is calculated, step by step

A chip starts as one rectangle (a *die*) printed many times on a round silicon wafer, 300 mm across.

1. **The wafer price.** The foundry charges per processed wafer. Rough 2024 to 2026 analyst estimates for TSMC ([Silicon Analysts](https://siliconanalysts.com/data/wafer-pricing), [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/tsmc-could-charge-up-to-usd45-000-for-1-6nm-wafers-rumors-allege-a-50-percent-increase-in-pricing-over-prior-gen-wafers)):
   - N16/N12: about $5,500
   - N7/N6: about $9,500
   - N5/N4: about $18,500 (marked low confidence by the source)
   - N3: about $20,000 to $22,000
   - N2: about $30,000
   - TSMC reportedly raises prices on advanced nodes by 3 to 10% a year.
2. **Dies per wafer.** A 300 mm wafer has about 70,700 mm² of area, but the round edge wastes some partial rectangles. The standard formula: dies = π·r²/A − π·d/√(2A).
   - For an 80 mm² die: 70,686/80 − 942/12.6 = 884 − 75 = **809 dies**.
3. **Yield.** Random defects kill some dies. A bigger die is more likely to contain a defect. With the Murphy yield model and an assumed 0.07 defects per cm²:
   - an 80 mm² die (0.8 cm²) yields about **95%**
   - a 300 mm² die yields about 81%
4. **Cost per good die** = wafer price ÷ (dies × yield).
   - 80 mm² on N6: $9,500 ÷ (809 × 0.95) = **about $12.40**.

From the script, cost per good die by node and size:
- **N7/N6:** 40 mm² $5.88, 80 mm² $12.42, 150 mm² $25.34, 300 mm² $59.27.
- **N5/N4:** 40 mm² $11.50, 80 mm² $24.37, 150 mm² $50.08, 300 mm² $118.81.
- **N3:** 40 mm² $12.53, 80 mm² $26.77, 150 mm² $55.75, 300 mm² $136.02.
- **N16/N12:** 40 mm² $3.38, 80 mm² $7.08, 150 mm² $14.24, 300 mm² $32.38.

## The surprising part: the die is a small slice of the cost

In our first-pass card (`07_competitors_and_goal.md`):
- **Memory:** 8 GB × $12 = **$96**
- **Board and power:** $35
- **Package:** $15
- **Die:** **$12** (80 mm², N6)

Moving that die from N6 to N3 would roughly double its cost (+$14), making the card about 9% more expensive. Moving it to N12 would save about $5 (−3%). **For a memory-bound chip, the node barely changes the unit cost.** That's the opposite of a compute-bound GPU or CPU, where the die dominates.

So the node decision is driven by other things.

## What actually decides the node for us

### 1. Can we get a GDDR7 PHY on it?

The PHY is the analog circuit that talks to the memory chips at 32 Gb/s per pin. Nobody designs one from scratch for a project like this; it's licensed IP from companies like Synopsys, Cadence or Rambus. GDDR7 PHYs are mostly offered on advanced nodes (N5/N4/N3 class). We need to check which nodes have a GDDR7 PHY available. If N6 doesn't, that alone may force N5/N4. **Open question.**

### 2. How much does the logic shrink?

Newer nodes pack more transistors per mm², so the same design gets smaller. But three parts of our chip shrink badly or not at all:
- **SRAM:** cell size barely improved from N5 to N3.
- **The PHYs and IO pads:** analog circuits and the bumps that connect to the package are set by physics and by bump spacing, not by transistor size.
- **The die edge:** a 128-bit GDDR7 interface needs a certain length of die edge for its pins (the *beachfront*). If the logic shrinks but the die can't get smaller than its edge needs, the shrink saves nothing.

Our chip is small logic plus a big memory interface, so a lot of the die may not shrink at all on a newer node. A cheaper node may be just as small in practice.

### 3. One-time costs (NRE)

These are paid once, however many chips you make:
- **Mask set:** the stencils used to print each layer. Mask costs rise steeply with node. Commonly quoted figures are millions of dollars at 7 nm and tens of millions at 3 nm. **We have not sourced these yet.**
- **IP licences:** the GDDR7 PHY and controller, PCIe, PLLs and standard cell libraries.
- **Design tools and engineering time,** and verification.

How NRE spreads over units:
- **$20M of NRE over 100,000 cards** = $200 per card. That's more than the whole unit cost.
- **Over 1,000,000 cards** = $20 per card.

So for a small company, **volume is a bigger lever than the node.** That's why D-008 reports NRE separately: unit cost is about the design, NRE is about the business.

### 4. Speed and power

Newer nodes switch faster and use less energy per operation. For a memory-bound chip, faster logic doesn't make tokens faster. Lower power helps a little with cooling and power-supply cost. Most of the energy goes into moving data to and from DRAM anyway.

## Other foundries

- **Samsung:** analyst estimates put its nodes about 20% below TSMC's: 5 nm about $13,000, 3 nm about $15,000.
- **Intel Foundry:** Intel 7 about $6,500. Intel 18A is newer, and pricing and PHY IP availability need checking.
- **GlobalFoundries:** 12LP+ about $4,200 per wafer. Cheap and mature, but no GDDR7 PHY that we know of.

All of these are estimates from the same analyst tracker. The trade-off when choosing a foundry: price vs IP availability vs yield maturity vs capacity. Advanced nodes are booked up by AI chips in 2026.

## Packaging, test and board

- **Package:** the die gets flipped onto a small substrate with solder bumps (*flip-chip BGA*). GDDR7's speed needs a good substrate. Old-style wire bonding won't do. Our guess: $10 to $20. Needs a source.
- **Test:** every die is tested on the wafer, and every packaged chip again. Test time costs money; built-in self-test (see `09_chip_infrastructure.md`) shortens it.
- **Board:** a PCIe card with power regulators, the 4 memory chips placed right next to the package, and a connector. GDDR7's speed makes the board layout demanding.
- **Binning:** chips that fail at full speed can be sold as slower versions. That's free yield recovery, though we'd probably skip it.

## Chiplets: not for us

Splitting a design into several small dies on one package (chiplets) helps when one die would be huge and yield badly, or when parts belong on different nodes. Our die is about 80 mm² with about 95% yield, so chiplets would only add packaging cost and complexity.

## Model-side levers, for completeness

These aren't manufacturing levers, but they cut **bytes per token**, and that's worth more than anything in this note:
- **MTP (multi-token prediction):** possibly 1.5 to 2× tokens per weight read.
- **Fewer bits per weight:** for example 3.5 bits average instead of 4.5, about 22% fewer bytes.

Both are in `07_competitors_and_goal.md` and need the golden model to check quality.

## What to find out

- **Which nodes have a licensable GDDR7 PHY** (and an LPDDR5X PHY as fallback), and roughly how big it is.
- **Mask set cost** by node, from a cited source.
- **Package and board cost** for a GDDR7 card.
- **A real area estimate for our die,** from synthesis of the matrix unit and SRAM plus PHY area from the IP vendors.
- **The RTX 5060 die size and node,** to compare our area properly.
