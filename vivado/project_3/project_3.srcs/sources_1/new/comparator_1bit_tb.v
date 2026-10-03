`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03.10.2026 14:27:45
// Design Name: 
// Module Name: comparator_1bit_tb
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


module comparator_1bit_tb;

reg A, B;
wire A_less_B;
wire A_equal_B;
wire A_greater_B;

comparator_1bit uut (
    .A(A),
    .B(B),
    .A_less_B(A_less_B),
    .A_equal_B(A_equal_B),
    .A_greater_B(A_greater_B)
);

initial begin

    $monitor("A=%b B=%b | A<B=%b A=B=%b A>B=%b",
             A, B, A_less_B, A_equal_B, A_greater_B);

    // Test case 1: A=0, B=0
    A = 0; B = 0;
    #10;

    // Test case 2: A=0, B=1
    A = 0; B = 1;
    #10;

    // Test case 3: A=1, B=0
    A = 1; B = 0;
    #10;

    // Test case 4: A=1, B=1
    A = 1; B = 1;
    #10;

    $finish;
end

endmodule