extern crate kdtree_na;
extern crate nalgebra;
extern crate num_traits;

use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use nalgebra::{storage::Storage, vector, DVector, Dim, Matrix, Norm, SimdComplexField, Vector2};
use num_traits::Zero;

use kdtree_na::KdTree;

pub struct EuclideanNormSquaredCounted<'a> {
    norm_count: &'a AtomicUsize,
    distance_count: &'a AtomicUsize,
}

impl<'a, X: SimdComplexField> Norm<X> for EuclideanNormSquaredCounted<'a> {
    #[inline]
    fn norm<R, C, S>(&self, m: &Matrix<X, R, C, S>) -> X::SimdRealField
    where
        R: Dim,
        C: Dim,
        S: Storage<X, R, C>,
    {
        self.norm_count.fetch_add(1, Ordering::SeqCst);
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
        self.distance_count.fetch_add(1, Ordering::SeqCst);
        m1.zip_fold(m2, X::SimdRealField::zero(), |acc, a, b| {
            let diff = a - b;
            acc + diff.simd_modulus_squared()
        })
    }
}

static POINT_A: (Vector2<f64>, usize) = (vector![0f64, 0f64], 0);
static POINT_B: (Vector2<f64>, usize) = (vector![1f64, 1f64], 1);
static POINT_C: (Vector2<f64>, usize) = (vector![2f64, 2f64], 2);
static POINT_D: (Vector2<f64>, usize) = (vector![3f64, 3f64], 3);

#[test]
fn it_works_with_static() {
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_capacity_static(capacity_per_node);

    let norm_count = AtomicUsize::new(0);
    let distance_count = AtomicUsize::new(0);
    let norm = EuclideanNormSquaredCounted {
        norm_count: &norm_count,
        distance_count: &distance_count,
    };

    kdtree.add(&POINT_A.0, POINT_A.1).unwrap();
    kdtree.add(&POINT_B.0, POINT_B.1).unwrap();
    kdtree.add(&POINT_C.0, POINT_C.1).unwrap();
    kdtree.add(&POINT_D.0, POINT_D.1).unwrap();

    kdtree.nearest(&POINT_A.0, 0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 0);

    kdtree.nearest(&POINT_A.0, 1, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    kdtree.nearest(&POINT_A.0, 2, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 4);

    kdtree.nearest(&POINT_A.0, 3, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&POINT_A.0, 4, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&POINT_A.0, 5, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&POINT_B.0, 4, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.within(&POINT_A.0, 0.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    kdtree.within(&POINT_B.0, 1.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 3);

    kdtree.within(&POINT_B.0, 2.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    let mut iter = kdtree.iter_nearest(&POINT_A.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 0);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 0);
}

#[test]
fn it_works_with_dynamic() {
    let dimensions = 2;
    let capacity_per_node = 2;
    let mut kdtree = KdTree::with_capacity_dynamic(2, capacity_per_node);

    let norm_count = AtomicUsize::new(0);
    let distance_count = AtomicUsize::new(0);
    let norm = EuclideanNormSquaredCounted {
        norm_count: &norm_count,
        distance_count: &distance_count,
    };

    let point_a = (
        DVector::from_iterator(dimensions, POINT_A.0.iter().cloned()),
        POINT_A.1,
    );
    let point_b = (
        DVector::from_iterator(dimensions, POINT_B.0.iter().cloned()),
        POINT_B.1,
    );
    let point_c = (
        DVector::from_iterator(dimensions, POINT_C.0.iter().cloned()),
        POINT_C.1,
    );
    let point_d = (
        DVector::from_iterator(dimensions, POINT_D.0.iter().cloned()),
        POINT_D.1,
    );

    kdtree.add(&point_a.0, point_a.1).unwrap();
    kdtree.add(&point_b.0, point_b.1).unwrap();
    kdtree.add(&point_c.0, point_c.1).unwrap();
    kdtree.add(&point_d.0, point_d.1).unwrap();

    kdtree.nearest(&point_a.0, 0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 0);

    kdtree.nearest(&point_a.0, 1, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    kdtree.nearest(&point_a.0, 2, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 4);

    kdtree.nearest(&point_a.0, 3, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&point_a.0, 4, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&point_a.0, 5, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.nearest(&point_b.0, 4, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    kdtree.within(&point_a.0, 0.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    kdtree.within(&point_b.0, 1.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 3);

    kdtree.within(&point_b.0, 2.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 6);

    let mut iter = kdtree.iter_nearest(&point_a.0, &norm).unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 0);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 2);

    iter.next().unwrap();
    assert_eq!(norm_count.swap(0, Ordering::SeqCst), 0);
    assert_eq!(distance_count.swap(0, Ordering::SeqCst), 0);
}
