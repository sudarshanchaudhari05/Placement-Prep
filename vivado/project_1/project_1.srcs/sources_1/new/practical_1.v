`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05.09.2026 14:53:15
// Design Name: 
// Module Name: practical_1
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

module boolean_behavioral(
    input  A,
    input  B,
    input  C,
    output reg Y
);

always @(*) begin
    Y = (A & B) | (~A & C);
end

endmodule
