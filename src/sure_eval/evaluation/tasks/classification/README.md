# Classification Task Route

Generic classification routes aligned `key<TAB>label` files through
`scoring/classify`. The scoring node needs a label spec that declares canonical
label ids, aliases, and optional numeric ids.

SER and GR use built-in compatibility specs so old artifacts keep the same
accuracy behavior while reporting the new pipeline trace.
