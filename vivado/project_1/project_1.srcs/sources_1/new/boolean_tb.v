`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 05.09.2026 15:03:22
// Design Name: 
// Module Name: boolean_tb
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

`timescale 1ns/1ps

module boolean_tb;

reg A, B, C;
wire Y;

boolean_behavioral DUT (
    .A(A),
    .B(B),
    .C(C),
    .Y(Y)
);

initial begin

    A = 0; B = 0; C = 0;
    #10;

    A = 0; B = 0; C = 1;
    #10;

    A = 0; B = 1; C = 0;
    #10;

    A = 0; B = 1; C = 1;
    #10;

    A = 1; B = 0; C = 0;
    #10;

    A = 1; B = 0; C = 1;
    #10;

    A = 1; B = 1; C = 0;
    #10;

    A = 1; B = 1; C = 1;
    #10;

    $finish;

end

endmodule