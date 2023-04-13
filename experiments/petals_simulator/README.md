# Petals Simulator

## TODO

- [X] Add timestamp field to `InferRequest`

### Server Selection on Client Side

It should be similar to how routing is done. Two things are important: the load
of the server, and the location. Both should be weighed equally right now, and
summed up. Then based on that, the server can make a routing decision.

## Testing

In order to run the tester code, which is inside `test` directory, we have to
take advantage of the Python `-m` flag, which is useful for working with
modules.

I added `__init__.py` files in both the main directory and the `test` directory,
which was needed, and this allows us to run, for example

```bash
python3 -m test.test_dht
```

in order to run the tests for the distributed hash table. Similar files need to
be written for *all* the components that are part of the simulator.

## Evaluation

### End-to-end Latency

```bash
cd experiments/petals_simulator
rm trace.json # in case that stale data exists
rm e2e_latency.txt # in case that stale data exists
python3 simulation.py
python3 evaluation/end2end_latency.py
```