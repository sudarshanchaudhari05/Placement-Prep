<CodeBlock language="tcl" editable>
## Input switches
## SW0 -> I0
set_property -dict { PACKAGE_PIN M20 IOSTANDARD LVCMOS33 } [get_ports { I0 }]

SW1 -> I1
set_property -dict { PACKAGE_PIN M19 IOSTANDARD LVCMOS33 } [get_ports { I1 }]
SW2 -> S
set_property -dict { PACKAGE_PIN L19 IOSTANDARD LVCMOS33 } [get_ports { S }]

Output LED
LED0 -> Y
set_property -dict { PACKAGE_PIN R14 IOSTANDARD LVCMOS33 } [get_ports { Y }]
</CodeBlock>