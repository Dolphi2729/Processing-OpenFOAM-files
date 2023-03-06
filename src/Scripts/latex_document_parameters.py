# Latex document parameters:
# T1 font encoded Palatino (ppl) medium weight (m)
# normal shape (n) 10pt font.

TEXT_WIDTH = 443.86319  # pt (\showthe\textwidth)
MATH_FONT = 'Palatino'  #\T1/ppl/m/n/10  (\showthe\font)
MATH_FONTSIZE = 5
TEXT_FONTSIZE = 2

ASPECT_RATIO = (3508. / 2480.)

# 0.005 is 0.5% gap between labels and axes (left and bottom edge)
#                        or axes and fig (top and bottom)
PRINT_GAP_Y = 0.005
PRINT_GAP_X = PRINT_GAP_Y / ASPECT_RATIO
# 0.015764 is taken by commas (obtained via inkscape)
Y_BOTTOM_LINE = 0.015764

TOP_VIEWPORT = 1.0 - PRINT_GAP_Y
RIGHT_VIEWPORT = 1.0 - PRINT_GAP_X

# LEFT_VIEWPORT and BOTTOM_VIEWPORT are obtain via make_test_plot()
LEFT_VIEWPORT = 0.129 + PRINT_GAP_X
BOTTOM_VIEWPORT = 0.080719505638848830 + Y_BOTTOM_LINE + PRINT_GAP_Y

PAD = 0.01
BPAD = 0.01