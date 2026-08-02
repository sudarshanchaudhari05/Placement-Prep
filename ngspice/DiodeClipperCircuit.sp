* Diode Clipper Circuit
* Input: 5 V peak, 1 kHz sine wave

Vin in 0 SIN(0 5 1k)

R1 in out 1k

D1 out 0 D4148

.model D4148 D

.tran 10u 5m

.control
run
plot v(in) v(out)
.endc

.end
