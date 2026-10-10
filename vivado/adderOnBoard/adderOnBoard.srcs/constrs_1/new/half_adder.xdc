## Input switches
set_property -dict { PACKAGE_PIN M20 IOSTANDARD LVCMOS33 } [get_ports { A }]
set_property -dict { PACKAGE_PIN M19 IOSTANDARD LVCMOS33 } [get_ports { B }]

## Output LEDs
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports { Sum }]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports { Carry }]
