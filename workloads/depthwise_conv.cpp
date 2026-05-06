#include <vector>
// Depthwise separable convolution (Phase 3, July)
//
// Standard conv: O(C_in * C_out * H * W * Kh * Kw) FLOPs
// Depthwise conv: O(C_in * H * W * Kh * Kw) FLOPs  — each channel independent
//
// Key property: much lower arithmetic intensity than GEMM.
// A 3x3 depthwise conv has ~9 MACs per output pixel vs hundreds for dense conv.
// This makes it strongly memory-bound — a good contrast point on the roofline.
//
// Used heavily in MobileNet, EfficientNet, and other mobile/edge models.
// d-Matrix and Cerebras care about this because their compute density
// means memory-bound ops are the bottleneck, not compute.

// TODO July: implement, benchmark, add to roofline plot
