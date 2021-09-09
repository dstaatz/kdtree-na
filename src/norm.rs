use nalgebra::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use nalgebra::{storage::Storage, Dim, Matrix, Norm, SimdComplexField};
use num_traits::Zero;

pub struct EuclideanNormSquared;

impl<X: SimdComplexField> Norm<X> for EuclideanNormSquared {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<X, R, C, S>) -> X::SimdRealField
    where
        R: Dim,
        C: Dim,
        S: Storage<X, R, C>,
    {
        m.norm_squared()
    }

    #[inline]
    fn metric_distance<R1, C1, S1, R2, C2, S2>(
        &self,
        m1: &Matrix<X, R1, C1, S1>,
        m2: &Matrix<X, R2, C2, S2>,
    ) -> X::SimdRealField
    where
        R1: Dim,
        C1: Dim,
        S1: Storage<X, R1, C1>,
        R2: Dim,
        C2: Dim,
        S2: Storage<X, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R1, R2> + SameNumberOfColumns<C1, C2>,
    {
        m1.zip_fold(m2, X::SimdRealField::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff.simd_modulus_squared()
        })
    }
}
