`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 24.09.2026 14:45:02
// Design Name: 
// Module Name: RCA
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


module RCA(
    input [3:0] a,
    input [3:0] b,
    input cin,
    output [3:0] s,
    output cout
);

wire c1, c2, c3;

FA f1(a[0], b[0], cin, s[0], c1);
FA f2(a[1], b[1], c1,  s[1], c2);
FA f3(a[2], b[2], c2,  s[2], c3);
FA f4(a[3], b[3], c3,  s[3], cout);

endmodule
