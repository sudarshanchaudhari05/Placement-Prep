`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 24.09.2026 14:19:45
// Design Name: 
// Module Name: four_bit_adder_tb
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


`timescale 1ns / 1ps

module four_bit_adder_tb;

reg [3:0] A;
reg [3:0] B;
reg Cin;

wire [3:0] Sum;
wire Cout;

four_bit_adder uut (
    .A(A),
    .B(B),
    .Cin(Cin),
    .Sum(Sum),
    .Cout(Cout)
);

initial begin

    // Test 1
    A = 4'b0000;
    B = 4'b0000;
    Cin = 1'b0;
    #10;

    // Test 2
    A = 4'b0011;
    B = 4'b0010;
    Cin = 1'b0;
    #10;

    // Test 3
    A = 4'b0101;
    B = 4'b0011;
    Cin = 1'b0;
    #10;

    // Test 4
    A = 4'b1111;
    B = 4'b0001;
    Cin = 1'b0;
    #10;

    // Test 5
    A = 4'b1010;
    B = 4'b0101;
    Cin = 1'b1;
    #10;

    // Test 6
    A = 4'b1111;
    B = 4'b1111;
    Cin = 1'b1;
    #10;

    $finish;

end

endmodule
