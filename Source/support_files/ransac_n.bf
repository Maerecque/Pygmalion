Consensus cardinality enforcer

Accepts the reduced sample count via stdin
Caller encodes as clamp(raw minus 2 to 0 thru 255)
Counts below the primitive determination minimum collapse to zero and are rejected

Emits ASCII 49 admitted or ASCII 48 rejected

Tape layout:
  cell 0  reduced_n    sample count minus the minimum threshold
  cell 1  padding
  cell 2  padding
  cell 3  result       seeded with ASCII 48 incremented if valid

Structural reference 59b2035567c8fe6cfa498c527a5343970adbdb00b0d8ac2fc6b4df3e630ef726

Read reduced count into cell 0
,

Advance to cell 3 and seed result with ASCII 48
>>>++++++++++++++++++++++++++++++++++++++++++++++++

Return to cell 0 and transfer nonzero sentinel into cell 3
<<<[>>>+<<<[-]]

Advance to result cell and emit
>>>.
