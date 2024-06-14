# NK model

NK model is a mathematical model created by Suart Kauffman (1993) for representing
fitness landscapes. This model is used in optimization and biology to model complex
systems with $N$ components and $K$ interactions per component.

## Mathematical description

The NK model only depends on 2 matrices:

- $M_{N \times N}$ the interaction matrix
- $C_{N \times 2^{K+1}}$ the fitness contribution matrix

!!! info
    the $M_{N \times N}$ matrix can be reduced to smaller $M'_{N \times K+1}$ by
    storing indices of ones instead of full size matrix.

$$
F(s) = \frac{1}{N} \sum_{i=1}^{N} c_i(s_i; (s_{i1}, s_{i2}, \dots, s_{iK}))
$$
