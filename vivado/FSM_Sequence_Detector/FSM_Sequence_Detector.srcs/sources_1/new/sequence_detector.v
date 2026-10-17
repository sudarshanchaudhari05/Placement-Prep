`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 17.10.2026 14:23:37
// Design Name: 
// Module Name: sequence_detector
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

module sequence_detector (
    input wire clk,
    input wire reset,
    input wire din,
    output reg detected
);

    // State encoding
    parameter S0 = 2'b00,
              S1 = 2'b01,
              S2 = 2'b10,
              S3 = 2'b11;

    reg [1:0] state;

    // FSM state transitions
    always @(posedge clk) begin
        if (reset) begin
            state    <= S0;
            detected <= 1'b0;
        end
        else begin
            detected <= 1'b0;

            case (state)
                S0: begin
                    if (din)
                        state <= S1;
                    else
                        state <= S0;
                end

                S1: begin
                    if (din)
                        state <= S1;
                    else
                        state <= S2;
                end

                S2: begin
                    if (din)
                        state <= S3;
                    else
                        state <= S0;
                end

                S3: begin
                    if (din) begin
                        detected <= 1'b1;
                        state <= S1;
                    end
                    else begin
                        state <= S2;
                    end
                end

                default: begin
                    state <= S0;
                    detected <= 1'b0;
                end
            endcase
        end
    end

endmodule

