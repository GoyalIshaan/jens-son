# 23. PCIe, explained, and which link our card uses

*Written 2026-11-18. Closes open item 11 (host interface) of the spec v1 plan. Blog-ready.*

## What PCIe is

- PCIe (PCI Express) is how cards talk to a computer: graphics cards, SSDs, network cards.
- **A lane** is two pairs of wires: one pair sends, the other receives, both at the same time.
- **A link** bundles 1, 2, 4, 8 or 16 lanes, written x1, x4, x16. More lanes means proportionally more bandwidth.
- **A generation** (Gen3, Gen4, Gen5 …) sets how fast each lane runs. Each generation roughly doubles the speed of the one before.

## The generations (usable bandwidth per direction)

- **Gen3** (2010): 8 GT/s per lane. x4 = 3.9 GB/s, x16 = 15.8 GB/s.
- **Gen4** (2017): 16 GT/s. x4 = **7.9 GB/s**, x16 = 31.5 GB/s.
- **Gen5** (2019): 32 GT/s. x4 = 15.8 GB/s, x16 = 63 GB/s.
- **Gen6** (2022, first products 2025): 64 GT/s, using 4-level signalling (PAM4) and error correction. x4 = 30.3 GB/s, x16 = 121 GB/s.
- **GT/s** means billions of transfers per second per lane. Gen3 to Gen5 lose about 1.5% to encoding (128b/130b), and the figures above already account for that.
- **Everything is backward compatible.** A Gen4 card works in a Gen3 slot at Gen3 speed. An x4 card works in an x16 slot. The two ends agree on the fastest speed and the most lanes both support.

## What our card actually sends over the link

**1. At startup, the weights.** 5.54 GB of main weights (4.46 GB of 4.5-bit matrices plus the 1.08 GB int8 embedding) and the 0.14 GB draft head, about 5.7 GB. Approximate load times:
- Gen3 x4: 1.4 s
- **Gen4 x4: 0.7 s**
- Gen5 x4: 0.36 s
- USB4 at the 3–4 GB/s it typically delivers to PCIe devices: 1.4–1.9 s

These are one-time costs, and the host can keep the weights cached so it only reloads after a reboot.

**2. While running, almost nothing:**
- The prompt goes in as token IDs, 4 bytes each. A 32K-token prompt is 128 KB, which takes 0.02 ms on Gen4 x4.
- Each generated token comes back as 4 bytes. Even at 300 tok/s that's about 1 KB/s.
- Optionally, the logits. 248,320 values × 2 bytes is about 0.5 MB per token. At 100 tok/s that's 50 MB/s, under 1% of Gen4 x4.
- Sampling happens on chip (note 19, item 8), so normally only tokens cross the link.

**3. Saving and restoring a long conversation** (moving the KV cache and DeltaNet state between host and card):
- At 128K context that's about 2.2 GB of KV plus about 50 MB of state: about 0.3 s on Gen4 x4.
- At 1M with int4 KV, about 9.2 GB: about 1.2 s.
- It only happens when switching conversations.

**So the link is never the bottleneck. Gen4 x4 is more than enough.**

## Ways to plug the card in

- **A normal card in an x16 slot, wired with only 4 lanes (recommended).**
  - Low-end graphics cards do the same thing (physically x16, electrically x4).
  - It fits any desktop and many small PCs.
  - **The power rule that matters:** a card that's physically x16 may draw **75 W from the slot**. A card that's physically x4 may draw only **25 W**. Our card should stay well under 75 W: prefill arithmetic is about 5–8 W (note 22), plus the GDDR7 chips and the rest of the die. So it needs **no extra power cable**.
- **M.2 (the slot SSDs use).** It carries up to PCIe x4 and is in most laptops and mini PCs. But it's built for SSD-sized power, and it's tiny (22 × 80 mm). Four GDDR7 chips, the die and their cooling won't fit, nor would the power. Fine for a future low-power version, not for this one.
- **OCuLink.** A cable carrying PCIe 4.0 x4 directly (OCuLink-2 can carry x8). It's used for external GPU boxes on mini PCs and handhelds. Full Gen4 x4 speed, if the host has the port. Enclosures cost about $119–309.
- **USB4 and Thunderbolt.** They can carry PCIe inside them ("tunnelling"), which lets laptops use external boxes. USB4 v1 is rated 40 Gb/s (5 GB/s), but PCIe usually gets less, since display and USB traffic share it. That's fine for us: the weights would load in under 2 s and running traffic is tiny. Not every USB4 port supports PCIe tunnelling, so check before relying on it.
- **CXL.** It runs on top of PCIe 5 and 6 and lets devices share memory with the CPU. Built for datacenter memory pooling. We don't need it.

## Cost on our side (what each choice costs the chip)

- **Die area for the PHY** (the high-speed transceiver circuits): bigger for more lanes and faster generations. Four lanes of Gen4 is a small, mature piece of IP that every major vendor sells and that's well proven at 6 nm-class nodes. Gen5 and Gen6 need more complex PHYs; Gen6 also needs 4-level signalling and error-correction logic.
- **IP licensing:** vendors don't publish prices. It's a one-time engineering cost (NRE), not a per-card cost, so it doesn't change the tok/s per $ score (D-008). Older, more common generations are cheaper and lower risk.
- **Board cost:** Gen4 x4 traces work on ordinary circuit-board material. Gen5 usually needs better board material, and sometimes retimer chips (signal regenerators, a few dollars to tens of dollars each) on long traces.
- **Compatibility:** Gen4 hosts have been everywhere since about 2020. A Gen4 card also works at Gen3 speed in older machines, where the weights load in 1.4 s instead of 0.7 s.

## The pick

- **PCIe Gen4 x4, on a physical x16 card (75 W slot power, no extra cable).**
- **Why not Gen5:** it would save 0.36 s at startup and nothing while running, but it costs more PHY area, better board material and more risk.
- **Why not x8 or x16:** more lanes mean more die area and pins, and they only speed up startup.
- **What the host needs:** any desktop slot that's physically x16 (even if it's wired x4), or an OCuLink or USB4 eGPU-style box for mini PCs and laptops.

## Sources

- PCIe generations, bandwidth, slot power (75 W for x16 graphics cards, 25 W for x4 cards), M.2, OCuLink, Thunderbolt, backward compatibility: https://en.wikipedia.org/wiki/PCI_Express (PCI-SIG specifications summarised there)
- OCuLink, USB4 v1/v2 bandwidth, PCIe tunnelling allocation, enclosure prices (checked 2026-11-13): https://tech-insider.org/oculink-vs-usb4-egpu-2026/
