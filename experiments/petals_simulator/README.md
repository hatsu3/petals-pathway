# Petals Simulator

## TODO

- [ ] Add timestamp field to `InferRequest`
- [ ] Server selection on the client side. 
- [ ] Servers should let DHT know about their load level

### Server Selection on Client Side

It should be similar to how routing is done. Two things are important: the load
of the server, and the location. Both should be weighed equally right now, and
summed up. Then based on that, the server can make a routing decision.
