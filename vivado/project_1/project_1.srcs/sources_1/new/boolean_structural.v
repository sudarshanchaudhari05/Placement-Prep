`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05.09.2026 15:02:07
// Design Name: 
// Module Name: boolean_structural
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


module boolean_structural(
    input  A,
    input  B,
    input  C,
    output Y
);

wire w1, w2, w3;

and G1(w1, A, B);
not G2(w2, A);
and G3(w3, w2, C);
or  G4(Y, w1, w3);

endmodule
