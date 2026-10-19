// Greedy curve thinning of region._spaced_indices — the same scalar arithmetic in double precision.
//
// A point is kept when its distance from the last kept point is at least `threshold`; the distance is
// sqrt(dx*dx + dy*dy) with dx, dy against the last kept point (built with -ffp-contract=off so the products are
// not fused). Where Python falls back to numpy's norm (non-finite distance, or |distance - threshold| <= tol),
// the kernel stops and returns that index; the caller decides it with numpy and resumes at the next index.
#include <cmath>
#include <cstdint>

extern "C" int64_t spaced_indices(const double* xy, int64_t n, int64_t start, int64_t last, double threshold,
                                  double tolerance, int64_t* keep, int64_t* n_keep) {
    double lx = xy[2 * last], ly = xy[2 * last + 1];
    int64_t m = 0;
    for (int64_t i = start; i < n; ++i) {
        const double x = xy[2 * i], y = xy[2 * i + 1];
        const double dx = x - lx, dy = y - ly;
        const double distance = std::sqrt(dx * dx + dy * dy);
        if (!std::isfinite(distance) || std::fabs(distance - threshold) <= tolerance) {
            *n_keep = m;
            return i;                       // numpy-norm fallback at i
        }
        if (distance >= threshold) {
            keep[m++] = i;
            lx = x; ly = y;
        }
    }
    *n_keep = m;
    return -1;
}
