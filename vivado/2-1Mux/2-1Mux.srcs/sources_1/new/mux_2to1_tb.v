`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10.10.2026 14:51:14
// Design Name: 
// Module Name: mux_2to1_tb
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////





module mux_2to1_tb;

reg I0, I1, S;
wire Y;

mux2to1 dut (
    .I0(I0),
    .I1(I1),
    .S(S),
    .Y(Y)
);

initial begin
    I0=0; I1=0; S=0; #10;
    I0=1; I1=0; S=0; #10;
    I0=0; I1=1; S=0; #10;
    I0=0; I1=1; S=1; #10;
    I0=1; I1=0; S=1; #10;
    I0=1; I1=1; S=1; #10;

    $finish;
end

endmodule
