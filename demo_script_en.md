# Demo narration (English) — three live demos for the defence

> Read time: about 3 minutes for demos 1 and 2, about 2 for demo 3.
> All numbers below are the ONE-EPISODE (seed 0) figures the window shows.
> The slide digits are 8-seed averages; say so once per demo and move on.

## Demo 1 — Results under Jamming (≈ 90 seconds)

Config: `adversary = jam`, `λ = 2.5`, `B_max = 1e6`, `n_X = 1e5`, `d = 55 km`,
ticks: always-pm, always-ent, greedy-0.045, block-aware, oracle.
Press Apply. Press Start.

SAY: "This is the key-limited scenario, lambda two point five, so the key
buffer runs dry and every policy is short of key. That is the only regime
where mode selection matters — at lambda one everything saturates at one
hundred percent and no controller can show anything."

SAY: "The amber bands are the jammed slots. Jamming raises the QBER of
whichever mode is running, to about fifty percent inside the band, so every
threshold policy can see it. Watch the mode lane: BlockAwareGreedy leaves PM
and runs ENT inside the bands — sixty switches in this episode — while
AlwaysPM sits still."

SAY: "The scoreboard reads zero point six seven seven five for AlwaysPM,
zero point seven for Greedy, zero point seven zero four three for AlwaysENT,
and zero point seven five three four for BlockAwareGreedy — the highest
realisable line. Oracle at zero point seven seven is the clairvoyant bound:
it sees the attack in advance, switches for free, and gets continuous key,
so it is dashed and labelled not achievable. The slide digits are the
eight-seed averages — BlockAware zero point seven zero three five, plus
eleven point nine percent over AlwaysPM and six point three percent over
AlwaysENT, taking about eighty percent of the Oracle headroom."

## Demo 2 — Results under the PNS Attack (≈ 90 seconds)

Config: everything at the window defaults — `adversary = pns`, `λ = 4`,
ticks: always-pm, always-ent, greedy-0.045, block-aware, decoy-0.100, oracle.
Press Apply. Press Start.

SAY: "This is the stealthy adversary. Photon-number splitting removes PM's
key without adding a single error — it blocks single-photon pulses and
relays multi-photon ones, so the security proof certifies less key while the
QBER stays at one percent. Look at the QBER panel: it is flat straight
through the red attack bands. That is the whole left half of the slide:
QBER is not enough, and a QBER-only controller misses the attack."

SAY: "In the mode lane, only DecoyWatch moves. It watches the decoy gain
ratio — the signal-to-decoy click ratio, which deviates by fifteen standard
deviations within a single block — and its QBER alarm is set to one, meaning
off: it reacts to the decoy ratio alone. Greedy and block-aware invert the
QBER to estimate the attack, read zero, and never leave PM: they are blind
not for lack of a model but because they observe the wrong quantity."

SAY: "The scoreboard: AlwaysPM, Greedy and BlockAware read zero point five
eight three six — bit-for-bit identical, zero switches. AlwaysENT reads
zero point six three eight three, since the attack cannot touch it. Decoy
reads zero point six eight zero seven with forty switches: the only line
that moves. Oracle is zero point seven zero six. The duty table on the slide
is a separate sweep over the adversary chain: the two fixed modes cross near
ten percent duty — ten point nine analytically — and that is where switching
gains most, plus eight point eight percent."

## Demo 3 — Block Length and the Policy Map (≈ 60 seconds)

SAY: "The left table is arithmetic, not a measurement. A PM block at fifty
five kilometres accumulates about sixty-nine thousand sifted bits per slot,
so a one-e-five block fills in one point four slots and a three-e-six block
in forty-three. The deadline is twenty slots. That ratio is the regime:
supply-limited at one-e-five, where goodput tracks key distilled; timing-
dominated at three-e-six, where key arrives in large late chunks."

LIVE: change one box — `n_X block = 1000000` — press Apply, press Start.
Ticks: always-pm, always-ent, decoy-0.100, oracle.

SAY: "Watch the buffer: it arrives in large late chunks instead of smoothly.
And note that each PM probe now discards a much bigger partial block —
equation five deletes the unfinished block on every switch — which is why
the decoy's advantage shrinks as the block grows, and why at three-e-six the
map says AlwaysENT everywhere. No single policy is optimal: the right choice
depends on distance, block length and the attack."
