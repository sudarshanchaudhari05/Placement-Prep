* PMOS with Exponential Gate Voltage
.include models.txt

VS S 0 1
VD D 0 0

* Exponential Voltage Source
* EXP(V1 V2 TD1 TAU1 TD2 TAU2)
VG G 0 EXP(1 0 1u 100n 5u 100n)

M1 D G S S P_50n W=1u L=90n

.tran 10n 10u

.control
run
plot v(G) i(VS)
.endc

.end