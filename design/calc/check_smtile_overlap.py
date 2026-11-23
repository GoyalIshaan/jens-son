"""Check the proposed SMTILE calendar without running or modifying RTL."""

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Calendar:
    positions: int
    interval: int
    subtract: int
    tree: tuple[int, ...]
    finalize: int
    state_write: int


CALENDARS = (
    Calendar(256, 8, 11, (85, 94, 103, 112, 121, 130, 139, 148, 157), 166, 175),
    Calendar(1024, 17, 17, (114, 123, 132, 141, 150, 159, 168, 177, 186), 196, 207),
)

FMA = 8
EXP = 64
QUARTER = (1 << 128) - 1
SCALAR = 1 << 127


def peak_live(start, last_use, interval):
    """Slots are reserved from allocation through the last use, inclusive."""
    clocks = defaultdict(int)
    for row in range(128):
        clocks[row * interval + start] += 1
        clocks[row * interval + last_use + 1] -= 1
    active = peak = 0
    for cycle in sorted(clocks):
        active += clocks[cycle]
        peak = max(peak, active)
    return peak


def check(calendar):
    positions, interval, subtract, tree, finalize, state_write = (
        calendar.positions, calendar.interval, calendar.subtract,
        calendar.tree, calendar.finalize, calendar.state_write,
    )
    beats = positions // 128
    scale_v = subtract + FMA + EXP
    p_write = scale_v + FMA
    last_p = scale_v + beats - 1
    last_sum = last_p + (FMA if beats > 4 else 0)
    assert subtract >= beats + FMA + 1, "running max is still in flight"
    assert len(tree) == 9 and tree[0] >= last_sum + 1, "lane sums are still in flight"
    for first, second in zip(tree, tree[1:]):
        assert second >= first + FMA + 1, "tree result is still in flight"
        assert second - (first + FMA) < interval, "tree level needs another buffer entry"
    assert finalize >= max(tree[-1] + FMA, scale_v) + 1
    assert state_write >= max(finalize + FMA + 1, p_write + beats)
    reservations = {}

    def reserve(cycle, resource, mask=1):
        key = (cycle % interval, resource)
        assert reservations.get(key, 0) & mask == 0, (positions, cycle, resource)
        reservations[key] = reservations.get(key, 0) | mask

    reserve(0, "SRAM read")
    for beat in range(beats):
        reserve(1 + beat, "SRAM read")
        reserve(1 + beat, "Q3", QUARTER)
        reserve(subtract + beat, "Q0", QUARTER)
        reserve(scale_v + beat, "Q1", QUARTER)
        if beat >= 4:
            reserve(scale_v + beat, "Q2", QUARTER)
        reserve(p_write + beat, "SRAM write")
    reserve(subtract, "Q3", SCALAR)
    reserve(tree[0], "Q0", QUARTER)
    reserve(tree[0], "Q1", QUARTER)
    for level, cycle in enumerate(tree[1:], start=1):
        reserve(cycle, "Q2", (1 << (256 >> level)) - 1)
    reserve(finalize, "Q3", SCALAR)
    reserve(state_write, "SRAM write")

    score_slots = peak_live(0, subtract + beats - 1, interval)
    scale_slots = peak_live(0, last_p, interval)
    sum_slots = peak_live(scale_v, tree[0], interval)
    row_slots = peak_live(0, state_write, interval)
    latency = state_write + 1
    print(f"{positions} positions: {interval} clocks/row, {latency} clocks first row, "
          f"{latency + 127 * interval} clocks for 128 rows")
    print(f"  peak slots: {score_slots} scores, {scale_slots} per-row V scales, "
          f"{sum_slots} lane sums, {row_slots} scalar contexts")


if __name__ == "__main__":
    for schedule in CALENDARS:
        check(schedule)
