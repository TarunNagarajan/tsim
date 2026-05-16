# tsim.

it's been way too long since i've worked on a long-term, hobby project. i'm going to build a high-performance fluid simulator.

## benchmarks (avx2 optimization)

| stage | before | after | speedup |
| :--- | :--- | :--- | :--- |
| advect | 1.49ms | 427.17micros | 3.48x |
| project | 11.33ms | 1.78ms | 6.36x |
