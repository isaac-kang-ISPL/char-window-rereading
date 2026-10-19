// Flooding of the seeded watershed with a distance term (compact watershed) — called by `segment._flood` through ctypes.
//
// A C++ port of scikit-image 0.26.0 `segmentation/_watershed_cy.pyx` (watershed_raveled · _euclid_dist) and
// `heap_general.pxi` · `heap_watershed.pxi` (Heapitem · smaller · heappush · heappop), for compactness > 0 and no
// watershed line only (BSD-3-Clause · compact_watershed.LICENSE). The same image and seeds give the same labels:
//   seeds are pushed as (elevation, age 0) (-inf with seeds_at_minus_inf, as in 0.25), a popped pixel takes the label of its
//   source, and each neighbor is pushed with the age raised by one as (elevation + compactness·distance, at least the
//   popped value). The heap's comparison, sift-up and sift-down order follow the original.
// Distances use the strides of the padded image, as in the original — with an F-order elevation the strides are (1, H),
// so the distance becomes one-dimensional (the original behavior).
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

struct Item { double value; int32_t age; int64_t index, source; };

inline bool smaller(const Item& a, const Item& b) {
    if (a.value != b.value) return a.value < b.value;
    return a.age < b.age;
}

struct Heap {
    std::vector<Item> a;
    void push(const Item& x) {
        a.push_back(x);
        size_t child = a.size() - 1;
        while (child > 0) {
            size_t parent = (child + 1) / 2 - 1;
            if (!smaller(a[child], a[parent])) break;
            std::swap(a[child], a[parent]);
            child = parent;
        }
    }
    Item pop() {                                   // take the first item, move the last item to the front, sift down
        Item out = a[0];
        a[0] = a.back();
        a.pop_back();
        size_t i = 0, smallest = 0;
        while (i * 2 + 1 < a.size()) {
            size_t l = i * 2 + 1, r = i * 2 + 2;
            if (smaller(a[l], a[i])) smallest = l;
            if (r < a.size() && smaller(a[r], a[smallest])) smallest = r;
            if (smallest == i) break;
            std::swap(a[i], a[smallest]);
            i = smallest;
        }
        return out;
    }
};

inline double euclid(int64_t p0, int64_t p1, const int64_t* strides, int64_t ndim) {
    double result = 0;
    for (int64_t i = 0; i < ndim; i++) {
        double curr = double(p0 / strides[i] - p1 / strides[i]);
        result += curr * curr;
        p0 %= strides[i];
        p1 %= strides[i];
    }
    return std::sqrt(result);
}

}  // namespace

extern "C" int compact_flood(const double* image, const uint8_t* mask, const int64_t* seeds, int64_t nseeds,
                             const int64_t* nhood, int64_t nneighbors, const int64_t* strides, int64_t ndim,
                             double compactness, int seeds_at_minus_inf, int32_t* output) {
    try {
        Heap heap;
        heap.a.reserve(size_t(nseeds) + 1024);
        const double inf = std::numeric_limits<double>::infinity();
        for (int64_t j = 0; j < nseeds; j++)
            heap.push(Item{seeds_at_minus_inf ? -inf : image[seeds[j]], 0, seeds[j], seeds[j]});
        int64_t age = 1;
        while (!heap.a.empty()) {
            Item elem = heap.pop();
            if (output[elem.index] && elem.index != elem.source) continue;     // pixel already filled from another neighbor
            output[elem.index] = output[elem.source];
            for (int64_t j = 0; j < nneighbors; j++) {
                int64_t nb = nhood[j] + elem.index;
                if (!mask[nb] || output[nb]) continue;
                age++;
                double value = image[nb] + compactness * euclid(nb, elem.source, strides, ndim);
                if (value < elem.value) value = elem.value;
                heap.push(Item{value, int32_t(age), nb, elem.source});
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
