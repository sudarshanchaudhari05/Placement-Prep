* PMOS ID vs VSG
.include models.txt

VS S 0 1
VD D 0 0
VG G 0 1

M1 D G S S P_50n W=1u L=90n

.dc VG 1 0 -0.01

.control
run
plot i(VS)
.endc

.end
