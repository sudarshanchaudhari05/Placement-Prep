`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 17.10.2026 14:24:41
// Design Name: 
// Module Name: tb_sequence_detector
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

module tb_sequence_detector;

    reg clk;
    reg reset;
    reg din;

    wire detected;

    reg [6:0] stream;
    integer i;
    integer count;

    // Design under test
    sequence_detector uut (
        .clk(clk),
        .reset(reset),
        .din(din),
        .detected(detected)
    );

    // Clock: 10 ns period
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test sequence
    initial begin
        reset = 1;
        din = 0;
        stream = 7'b1011011;
        count = 0;

        repeat (2) @(negedge clk);
        reset = 0;

        $display("Input  Detected");

        for (i = 6; i >= 0; i = i - 1) begin
            @(negedge clk);
            din = stream[i];

            @(posedge clk);
            #1;

            $display("  %b       %b", din, detected);

            if (detected)
                count = count + 1;
        end

        $display("Total detections = %0d", count);

        if (count == 2)
            $display("TEST PASSED");
        else
            $display("TEST FAILED");

        #10;
        $finish;
    end

endmodule

