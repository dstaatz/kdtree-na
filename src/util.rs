use nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, Norm, OVector, RealField, Vector,
};

pub fn distance_to_space<X, D, S1, S2, S3>(
    p1: &Vector<X, D, S1>,
    min_bounds: &Vector<X, D, S2>,
    max_bounds: &Vector<X, D, S3>,
    norm: &impl Norm<X>,
) -> X
where
    X: RealField,
    D: Dim,
    S1: Storage<X, D>,
    S2: Storage<X, D>,
    S3: Storage<X, D>,
    DefaultAllocator: Allocator<X, D>,
{
    let (rows, cols) = p1.shape_generic();
    let p2 = OVector::from_fn_generic(rows, cols, |i, _| {
        if p1[i] > max_bounds[i] {
            max_bounds[i].clone()
        } else if p1[i] < min_bounds[i] {
            min_bounds[i].clone()
        } else {
            p1[i].clone()
        }
    });
    norm.metric_distance(&p1, &p2)
}

#[cfg(test)]
mod tests {
    use super::distance_to_space;
    use crate::norm::EuclideanNormSquared;
    use std::f64::{INFINITY, NEG_INFINITY};

    #[test]
    fn test_normal_distance_to_space() {
        let dis = distance_to_space(
            &[0.0, 0.0].into(),
            &[1.0, 1.0].into(),
            &[2.0, 2.0].into(),
            &EuclideanNormSquared,
        );
        assert_eq!(dis, 2.0);
    }

    #[test]
    fn test_distance_outside_inf() {
        let dis = distance_to_space(
            &[0.0, 0.0].into(),
            &[1.0, 1.0].into(),
            &[INFINITY, INFINITY].into(),
            &EuclideanNormSquared,
        );
        assert_eq!(dis, 2.0);
    }

    #[test]
    fn test_distance_inside_inf() {
        let dis = distance_to_space(
            &[2.0, 2.0].into(),
            &[NEG_INFINITY, NEG_INFINITY].into(),
            &[INFINITY, INFINITY].into(),
            &EuclideanNormSquared,
        );
        assert_eq!(dis, 0.0);
    }

    #[test]
    fn test_distance_inside_normal() {
        let dis = distance_to_space(
            &[2.0, 2.0].into(),
            &[0.0, 0.0].into(),
            &[3.0, 3.0].into(),
            &EuclideanNormSquared,
        );
        assert_eq!(dis, 0.0);
    }

    #[test]
    fn distance_to_half_space() {
        let dis = distance_to_space(
            &[-2.0, 0.0].into(),
            &[0.0, NEG_INFINITY].into(),
            &[INFINITY, INFINITY].into(),
            &EuclideanNormSquared,
        );
        assert_eq!(dis, 4.0);
    }
}
