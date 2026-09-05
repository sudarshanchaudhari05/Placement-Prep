`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05.09.2026 14:54:47
// Design Name: 
// Module Name: boolean_dataflow
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

module boolean_dataflow(
    input  A,
    input  B,
    input  C,
    output Y
);

assign Y = (A & B) | (~A & C);

endmodule
