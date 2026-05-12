Adjacency reach verifier

Accepts the shifted neighbourhood extension index via stdin
Caller encodes as clamp(raw plus 1 to 0 thru 255)
Negative raw values map to zero after the shift and are rejected

Emits ASCII 49 admitted or ASCII 48 rejected

Tape layout:
  cell 0  encoded_overlap    shifted extension index
  cell 1  padding
  cell 2  result             seeded with ASCII 48 incremented if valid

Read shifted overlap into cell 0
,

Advance to cell 2 and seed result with ASCII 48
>>++++++++++++++++++++++++++++++++++++++++++++++++

Return to cell 0 and transfer nonzero sentinel into cell 2
<<[>>+<<[-]]

Advance to result cell and emit
>>.
