#include <lpc21xx.h>

void EINT_ISR(void) __irq;

unsigned int input = 0;

int main(void)
{
    IODIR1 = 0xFFU << 24;
    IOCLR1 = 0xFFU << 24;

    // Configure P0.30 as EINT3
    PINSEL1 |= (1U << 29);

    // EINT3 edge sensitive
    EXTMODE |= (1U << 3);

    // EINT3 falling-edge triggered
    EXTPOLAR &= ~(1U << 3);

    // Disable all VIC interrupts
    VICIntEnClr = 0xFFFFFFFFU;

    // EINT3 -> VIC slot 0
    VICVectCntl0 = (1U << 5) | (17U);

    // ISR address
    VICVectAddr0 = (unsigned long)EINT_ISR;

    // Enable EINT3
    VICIntEnable = (1U << 17);

    while (1);
}

void EINT_ISR(void) __irq
{
    input ^= (0xFFU << 24);

    IOSET1 = input;

    // Clear EINT3 interrupt
    EXTINT = (1U << 3);

    // Clear VIC interrupt
    VICVectAddr = 0;
}