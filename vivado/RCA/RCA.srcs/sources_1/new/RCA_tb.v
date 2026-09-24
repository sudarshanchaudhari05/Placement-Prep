
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 24.09.2026 14:48:44
// Design Name: 
// Module Name: RCA_tb
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

module RCA_tb;

reg [3:0] a;
reg [3:0] b;
reg cin;

wire [3:0] s;
wire cout;

RCA DUT(a, b, cin, s, cout);

initial begin

    a = 4'b0000; b = 4'b0000; cin = 0;
    #10 a = 4'b0011; b = 4'b0010; cin = 0;
    #10 a = 4'b0101; b = 4'b0011; cin = 0;
    #10 a = 4'b1111; b = 4'b0001; cin = 0;
    #10 a = 4'b1010; b = 4'b0101; cin = 1;
    #10 a = 4'b1111; b = 4'b1111; cin = 1;

    #10 $finish;

end

endmodule
