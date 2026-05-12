Spatial resolution sentinel

Accepts the centesimal encoding of the cell edge magnitude via stdin
Caller encodes as clamp(round(raw times 100) to 0 thru 255)
A zero input signals subresolution magnitude and is rejected

Emits ASCII 49 admitted or ASCII 48 rejected

Tape layout:
  cell 0  encoded_res    centesimal magnitude
  cell 1  result         incremented if magnitude is nonzero

Read encoded resolution into cell 0
,

Transfer nonzero sentinel of cell 0 into cell 1
[>+<[-]]

Advance to cell 1 and seed with ASCII 48 then increment if nonzero
>++++++++++++++++++++++++++++++++++++++++++++++++.
