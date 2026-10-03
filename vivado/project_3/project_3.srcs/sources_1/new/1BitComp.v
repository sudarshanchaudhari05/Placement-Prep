`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.10.2026 14:22:50
// Design Name: 
// Module Name: 1BitComp
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


module comparator_1bit (
    input  A,
    input  B,
    output A_less_B,
    output A_equal_B,
    output A_greater_B
);

assign A_less_B    = (~A) & B;
assign A_equal_B   = ~(A ^ B);
assign A_greater_B = A & (~B);

endmodule
