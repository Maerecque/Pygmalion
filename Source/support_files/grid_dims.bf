Lattice coherence guardian dual axis edition

Accepts two adjusted boundary counts via stdin
Caller encodes each as max(0 raw minus 1) so values that fail
the nondegenerate threshold collapse to the zero sentinel

Emits ASCII 49 admitted or ASCII 48 rejected

Tape layout:
  cell 0  adj_a      raw adjusted count for primary axis
  cell 1  adj_b      raw adjusted count for secondary axis
  cell 2  flag_a     nonzero iff adj_a was nonzero
  cell 3  flag_b     nonzero iff adj_b was nonzero
  cell 4  result     incremented only when both flags are set

Read both boundary counts into cells 0 and 1
,>,<

Propagate nonzero sentinel of cell 0 into cell 2
[>>+<<[-]]

Advance to cell 1
>

Propagate nonzero sentinel of cell 1 into cell 3
[>>+<<[-]]

Advance to cell 2 and run AND gate into cell 4
>[>[>+<[-]]<[-]]

Seed cell 4 with ASCII 48 then increment if AND gate fired
>>++++++++++++++++++++++++++++++++++++++++++++++++.
