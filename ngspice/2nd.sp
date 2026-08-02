* Example : Pulse sourse in NGSpice

Vin in 0 PULSE(0 5 1ms 1us 1us 5ms 10ms)
R1 in out 1k
C1 out 0 1u

.tran 0.1ms 30ms

.control
run
plot v(in) v(out)
.endc

.end