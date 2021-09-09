use std::collections::BinaryHeap;

use nalgebra::{
    allocator::Allocator, storage::Storage, Const, DefaultAllocator, Dim, DimName, Dynamic, Norm,
    OVector, RealField, Vector,
};

use crate::heap_element::HeapElement;
use crate::util;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct KdTree<X, T, D: Dim>
where
    DefaultAllocator: Allocator<X, D>,
{
    // node
    left: Option<Box<KdTree<X, T, D>>>,
    right: Option<Box<KdTree<X, T, D>>>,
    // common
    dimensions: D,
    capacity: usize,
    size: usize,
    min_bounds: OVector<X, D>,
    max_bounds: OVector<X, D>,
    // stem
    split_value: Option<X>,
    split_dimension: Option<usize>,
    // leaf
    points: Option<Vec<OVector<X, D>>>,
    bucket: Option<Vec<T>>,
}

#[derive(Debug, PartialEq)]
pub enum ErrorKind {
    WrongDimension,
    NonFiniteCoordinate,
    ZeroCapacity,
}

impl<X: RealField + Copy, D: DimName, T> KdTree<X, T, D>
where
    DefaultAllocator: Allocator<X, D>,
{
    /// Create a new KD tree, specifying the dimension size of each point
    ///
    /// Statically determines dimension size from generic `D`
    pub fn new_static() -> Self {
        KdTree::new_generic(D::name())
    }

    /// Create a new KD tree, specifying the dimension size of each point and the capacity of leaf nodes
    ///
    /// Statically determines dimension size from generic `D`
    pub fn with_capacity_static(capacity: usize) -> Self {
        KdTree::with_capacity_generic(D::name(), capacity)
    }
}

impl<X: RealField + Copy, T> KdTree<X, T, Dynamic>
where
    DefaultAllocator: Allocator<X, Dynamic>,
{
    /// Create a new KD tree, specifying the dimension size of each point
    ///
    /// Dynamically determine the dimension size from argument `dimensions`
    pub fn new_dynamic(dimensions: usize) -> Self {
        KdTree::new_generic(Dynamic::new(dimensions))
    }

    /// Create a new KD tree, specifying the dimension size of each point and the capacity of leaf nodes
    ///
    /// Dynamically determine the dimension size from argument `dimensions`
    pub fn with_capacity_dynamic(dimensions: usize, capacity: usize) -> Self {
        KdTree::with_capacity_generic(Dynamic::new(dimensions), capacity)
    }
}

impl<X: RealField + Copy, D: Dim, T> KdTree<X, T, D>
where
    DefaultAllocator: Allocator<X, D>,
{
    /// Create a new KD tree, specifying the dimension size of each point
    fn new_generic(dims: D) -> Self {
        KdTree::with_capacity_generic(dims, 2_usize.pow(4))
    }

    /// Create a new KD tree, specifying the dimension size of each point and the capacity of leaf nodes
    fn with_capacity_generic(dims: D, capacity: usize) -> Self {
        let min_bounds = OVector::repeat_generic(dims, Const::<1>, X::max_value());
        let max_bounds = OVector::repeat_generic(dims, Const::<1>, X::min_value());
        KdTree {
            left: None,
            right: None,
            dimensions: dims,
            capacity,
            size: 0,
            min_bounds,
            max_bounds,
            split_value: None,
            split_dimension: None,
            points: Some(vec![]),
            bucket: Some(vec![]),
        }
    }
}

impl<X: RealField + Copy, D: Dim, T: PartialEq> KdTree<X, T, D>
where
    DefaultAllocator: Allocator<X, D>,
{
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn nearest<S>(
        &self,
        point: &Vector<X, D, S>,
        num: usize,
        norm: &impl Norm<X>,
    ) -> Result<Vec<(X, &T)>, ErrorKind>
    where
        S: Storage<X, D>,
    {
        if let Err(err) = self.check_point(point) {
            return Err(err);
        }
        let num = std::cmp::min(num, self.size);
        if num == 0 {
            return Ok(vec![]);
        }
        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<X, &T>>::new();
        pending.push(HeapElement {
            distance: X::zero(),
            element: self,
        });
        while !pending.is_empty()
            && (evaluated.len() < num
                || (-pending.peek().unwrap().distance <= evaluated.peek().unwrap().distance))
        {
            self.nearest_step(
                point,
                num,
                X::max_value(),
                norm,
                &mut pending,
                &mut evaluated,
            );
        }
        Ok(evaluated
            .into_sorted_vec()
            .into_iter()
            .take(num)
            .map(Into::into)
            .collect())
    }

    pub fn within<S>(
        &self,
        point: &Vector<X, D, S>,
        radius: X,
        norm: &impl Norm<X>,
    ) -> Result<Vec<(X, &T)>, ErrorKind>
    where
        S: Storage<X, D>,
    {
        if let Err(err) = self.check_point(point) {
            return Err(err);
        }
        if self.size == 0 {
            return Ok(vec![]);
        }
        let mut pending = BinaryHeap::new();
        let mut evaluated = BinaryHeap::<HeapElement<X, &T>>::new();
        pending.push(HeapElement {
            distance: X::zero(),
            element: self,
        });
        while !pending.is_empty() && (-pending.peek().unwrap().distance <= radius) {
            self.nearest_step(point, self.size, radius, norm, &mut pending, &mut evaluated);
        }
        Ok(evaluated
            .into_sorted_vec()
            .into_iter()
            .map(Into::into)
            .collect())
    }

    fn nearest_step<'b, S>(
        &self,
        point: &Vector<X, D, S>,
        num: usize,
        max_dist: X,
        norm: &impl Norm<X>,
        pending: &mut BinaryHeap<HeapElement<X, &'b Self>>,
        evaluated: &mut BinaryHeap<HeapElement<X, &'b T>>,
    ) where
        S: Storage<X, D>,
    {
        let mut curr = &*pending.pop().unwrap().element;
        debug_assert!(evaluated.len() <= num);
        let evaluated_dist = if evaluated.len() == num {
            // We only care about the nearest `num` points, so if we already have `num` points,
            // any more point we add to `evaluated` must be nearer then one of the point already in
            // `evaluated`.
            max_dist.min(evaluated.peek().unwrap().distance)
        } else {
            max_dist
        };

        while !curr.is_leaf() {
            let candidate;
            if curr.belongs_in_left(point) {
                candidate = curr.right.as_ref().unwrap();
                curr = curr.left.as_ref().unwrap();
            } else {
                candidate = curr.left.as_ref().unwrap();
                curr = curr.right.as_ref().unwrap();
            }
            let candidate_to_space =
                util::distance_to_space(&point, &candidate.min_bounds, &candidate.max_bounds, norm);
            if candidate_to_space <= evaluated_dist {
                pending.push(HeapElement {
                    distance: candidate_to_space * -X::one(),
                    element: &**candidate,
                });
            }
        }

        let points = curr.points.as_ref().unwrap().iter();
        let bucket = curr.bucket.as_ref().unwrap().iter();
        let iter = points.zip(bucket).map(|(p, d)| HeapElement {
            distance: norm.metric_distance(&point, p),
            element: d,
        });
        for element in iter {
            if element <= max_dist {
                if evaluated.len() < num {
                    evaluated.push(element);
                } else if element < *evaluated.peek().unwrap() {
                    evaluated.pop();
                    evaluated.push(element);
                }
            }
        }
    }

    pub fn iter_nearest<'a, 'b, S, N>(
        &'b self,
        point: &'a Vector<X, D, S>,
        norm: &'a N,
    ) -> Result<NearestIter<'a, 'b, X, T, D, N, S>, ErrorKind>
    where
        S: 'a + Storage<X, D>,
        N: Norm<X>,
    {
        if let Err(err) = self.check_point(point) {
            return Err(err);
        }
        let mut pending = BinaryHeap::new();
        let evaluated = BinaryHeap::<HeapElement<X, &T>>::new();
        pending.push(HeapElement {
            distance: X::zero(),
            element: self,
        });
        Ok(NearestIter {
            point,
            pending,
            evaluated,
            norm,
        })
    }

    pub fn iter_nearest_mut<'a, 'b, S, N>(
        &'b mut self,
        point: &'a Vector<X, D, S>,
        norm: &'a N,
    ) -> Result<NearestIterMut<'a, 'b, X, T, D, N, S>, ErrorKind>
    where
        S: 'a + Storage<X, D>,
        N: Norm<X>,
    {
        if let Err(err) = self.check_point(point) {
            return Err(err);
        }
        let mut pending = BinaryHeap::new();
        let evaluated = BinaryHeap::<HeapElement<X, &mut T>>::new();
        pending.push(HeapElement {
            distance: X::zero(),
            element: self,
        });
        Ok(NearestIterMut {
            point,
            pending,
            evaluated,
            norm,
        })
    }

    pub fn add<S>(&mut self, point: &Vector<X, D, S>, data: T) -> Result<(), ErrorKind>
    where
        S: Storage<X, D>,
    {
        if self.capacity == 0 {
            return Err(ErrorKind::ZeroCapacity);
        }
        if let Err(err) = self.check_point(point) {
            return Err(err);
        }
        self.add_unchecked(point, data)
    }

    fn add_unchecked<S>(&mut self, point: &Vector<X, D, S>, data: T) -> Result<(), ErrorKind>
    where
        S: Storage<X, D>,
    {
        if self.is_leaf() {
            self.add_to_bucket(point, data);
            return Ok(());
        }
        self.extend(point);
        self.size += 1;
        let next = if self.belongs_in_left(point) {
            self.left.as_mut()
        } else {
            self.right.as_mut()
        };
        next.unwrap().add_unchecked(point, data)
    }

    fn add_to_bucket<S>(&mut self, point: &Vector<X, D, S>, data: T)
    where
        S: Storage<X, D>,
    {
        self.extend(point);
        let mut points = self.points.take().unwrap();
        let mut bucket = self.bucket.take().unwrap();
        points.push(point.clone_owned());
        bucket.push(data);
        self.size += 1;
        if self.size > self.capacity {
            self.split(points, bucket);
        } else {
            self.points = Some(points);
            self.bucket = Some(bucket);
        }
    }

    pub fn remove<S>(&mut self, point: &Vector<X, D, S>, data: &T) -> Result<usize, ErrorKind>
    where
        S: Storage<X, D>,
    {
        let mut removed = 0;
        if let Err(err) = self.check_point(point) {
            return Err(err);
        }
        if let (Some(mut points), Some(mut bucket)) = (self.points.take(), self.bucket.take()) {
            while let Some(p_index) = points.iter().position(|x| x == point) {
                if &bucket[p_index] == data {
                    points.remove(p_index);
                    bucket.remove(p_index);
                    removed += 1;
                    self.size -= 1;
                }
            }
            self.points = Some(points);
            self.bucket = Some(bucket);
        } else {
            if let Some(right) = self.right.as_mut() {
                let right_removed = right.remove(point, data)?;
                if right_removed > 0 {
                    self.size -= right_removed;
                    removed += right_removed;
                }
            }
            if let Some(left) = self.left.as_mut() {
                let left_removed = left.remove(point, data)?;
                if left_removed > 0 {
                    self.size -= left_removed;
                    removed += left_removed;
                }
            }
        }
        Ok(removed)
    }

    fn split(&mut self, mut points: Vec<OVector<X, D>>, mut bucket: Vec<T>) {
        let mut max = X::zero();
        for dim in 0..self.dimensions.value() {
            let diff = self.max_bounds[dim] - self.min_bounds[dim];
            // if !diff.is_nan() && diff > max {
            if diff > max && max < diff {
                // Test that both directions give the same result
                max = diff;
                self.split_dimension = Some(dim);
            }
        }
        match self.split_dimension {
            None => {
                self.points = Some(points);
                self.bucket = Some(bucket);
                return;
            }
            Some(dim) => {
                let min = self.min_bounds[dim];
                let max = self.max_bounds[dim];
                self.split_value = Some(min + (max - min) / (X::one() + X::one()));
            }
        };
        let mut left = Box::new(KdTree::with_capacity_generic(
            self.dimensions,
            self.capacity,
        ));
        let mut right = Box::new(KdTree::with_capacity_generic(
            self.dimensions,
            self.capacity,
        ));
        while !points.is_empty() {
            let point = points.swap_remove(0);
            let data = bucket.swap_remove(0);
            if self.belongs_in_left(&point) {
                left.add_to_bucket(&point, data);
            } else {
                right.add_to_bucket(&point, data);
            }
        }
        self.left = Some(left);
        self.right = Some(right);
    }

    fn belongs_in_left<S>(&self, point: &Vector<X, D, S>) -> bool
    where
        S: Storage<X, D>,
    {
        point[self.split_dimension.unwrap()] < self.split_value.unwrap()
    }

    fn extend<S>(&mut self, point: &Vector<X, D, S>)
    where
        S: Storage<X, D>,
    {
        let min = self.min_bounds.iter_mut();
        let max = self.max_bounds.iter_mut();
        for ((l, h), v) in min.zip(max).zip(point.iter()) {
            if v < l {
                *l = *v
            }
            if v > h {
                *h = *v
            }
        }
    }

    fn is_leaf(&self) -> bool {
        self.bucket.is_some()
            && self.points.is_some()
            && self.split_value.is_none()
            && self.split_dimension.is_none()
            && self.left.is_none()
            && self.right.is_none()
    }

    fn check_point<S>(&self, point: &Vector<X, D, S>) -> Result<(), ErrorKind>
    where
        S: Storage<X, D>,
    {
        if self.dimensions != point.shape_generic().0 {
            return Err(ErrorKind::WrongDimension);
        }
        for n in point.iter() {
            if !n.is_finite() {
                return Err(ErrorKind::NonFiniteCoordinate);
            }
        }
        Ok(())
    }
}

pub struct NearestIter<'a, 'b, X, T, D, N, S>
where
    X: 'a + 'b + RealField,
    T: 'b + PartialEq,
    D: Dim,
    N: Norm<X>,
    S: 'a + Storage<X, D>,
    DefaultAllocator: Allocator<X, D>,
{
    point: &'a Vector<X, D, S>,
    pending: BinaryHeap<HeapElement<X, &'b KdTree<X, T, D>>>,
    evaluated: BinaryHeap<HeapElement<X, &'b T>>,
    norm: &'a N,
}

impl<'a, 'b, X, T, D, N, S> Iterator for NearestIter<'a, 'b, X, T, D, N, S>
where
    X: RealField + Copy,
    T: 'b + PartialEq,
    D: Dim,
    N: Norm<X>,
    S: 'a + Storage<X, D>,
    DefaultAllocator: Allocator<X, D>,
{
    type Item = (X, &'b T);
    fn next(&mut self) -> Option<(X, &'b T)> {
        use util::distance_to_space;

        let norm = self.norm;
        let point = self.point;
        while !self.pending.is_empty()
            && (self
                .evaluated
                .peek()
                .map_or(X::max_value(), |x| -x.distance)
                >= -self.pending.peek().unwrap().distance)
        {
            let mut curr = &*self.pending.pop().unwrap().element;
            while !curr.is_leaf() {
                let candidate;
                if curr.belongs_in_left(&point) {
                    candidate = curr.right.as_ref().unwrap();
                    curr = curr.left.as_ref().unwrap();
                } else {
                    candidate = curr.left.as_ref().unwrap();
                    curr = curr.right.as_ref().unwrap();
                }
                self.pending.push(HeapElement {
                    distance: -distance_to_space(
                        point,
                        &candidate.min_bounds,
                        &candidate.max_bounds,
                        norm,
                    ),
                    element: &**candidate,
                });
            }
            let points = curr.points.as_ref().unwrap().iter();
            let bucket = curr.bucket.as_ref().unwrap().iter();
            self.evaluated
                .extend(points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: -norm.metric_distance(&point, p),
                    element: d,
                }));
        }
        self.evaluated.pop().map(|x| (-x.distance, x.element))
    }
}

pub struct NearestIterMut<'a, 'b, X, T, D, N, S>
where
    X: 'a + 'b + RealField,
    T: 'b + PartialEq,
    D: Dim,
    N: Norm<X>,
    S: 'a + Storage<X, D>,
    DefaultAllocator: Allocator<X, D>,
{
    point: &'a Vector<X, D, S>,
    pending: BinaryHeap<HeapElement<X, &'b mut KdTree<X, T, D>>>,
    evaluated: BinaryHeap<HeapElement<X, &'b mut T>>,
    norm: &'a N,
}

impl<'a, 'b, X, T, D, N, S> Iterator for NearestIterMut<'a, 'b, X, T, D, N, S>
where
    X: RealField + Copy,
    T: 'b + PartialEq,
    D: Dim,
    N: Norm<X>,
    S: 'a + Storage<X, D>,
    DefaultAllocator: Allocator<X, D>,
{
    type Item = (X, &'b mut T);
    fn next(&mut self) -> Option<(X, &'b mut T)> {
        use util::distance_to_space;

        let norm = self.norm;
        let point = self.point;
        while !self.pending.is_empty()
            && (self
                .evaluated
                .peek()
                .map_or(X::max_value(), |x| -x.distance)
                >= -self.pending.peek().unwrap().distance)
        {
            let mut curr = &mut *self.pending.pop().unwrap().element;
            while !curr.is_leaf() {
                let candidate;
                if curr.belongs_in_left(&point) {
                    candidate = curr.right.as_mut().unwrap();
                    curr = curr.left.as_mut().unwrap();
                } else {
                    candidate = curr.left.as_mut().unwrap();
                    curr = curr.right.as_mut().unwrap();
                }
                self.pending.push(HeapElement {
                    distance: -distance_to_space(
                        point,
                        &candidate.min_bounds,
                        &candidate.max_bounds,
                        norm,
                    ),
                    element: &mut **candidate,
                });
            }
            let points = curr.points.as_ref().unwrap().iter();
            let bucket = curr.bucket.as_mut().unwrap().iter_mut();
            self.evaluated
                .extend(points.zip(bucket).map(|(p, d)| HeapElement {
                    distance: -norm.metric_distance(&point, p),
                    element: d,
                }));
        }
        self.evaluated.pop().map(|x| (-x.distance, x.element))
    }
}

impl std::error::Error for ErrorKind {
    fn description(&self) -> &str {
        match *self {
            ErrorKind::WrongDimension => "wrong dimension",
            ErrorKind::NonFiniteCoordinate => "non-finite coordinate",
            ErrorKind::ZeroCapacity => "zero capacity",
        }
    }
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "KdTree error: {}", self.to_string())
    }
}

#[cfg(test)]
mod tests {
    mod svector {
        extern crate rand;

        use nalgebra::{Const, SVector};

        use super::super::KdTree;

        const DIMS: usize = 2;

        fn random_point() -> (SVector<f64, DIMS>, i32) {
            (SVector::new_random(), rand::random())
        }

        #[test]
        fn it_has_default_capacity() {
            let tree: KdTree<f64, i32, Const<DIMS>> = KdTree::new_static();
            assert_eq!(tree.capacity, 2_usize.pow(4));
        }

        #[test]
        fn it_can_be_cloned() {
            let mut tree = KdTree::new_static();
            let (pos, data) = random_point();
            tree.add(&pos, data).unwrap();
            let mut cloned_tree = tree.clone();
            cloned_tree.add(&pos, data).unwrap();
            assert_eq!(tree.size(), 1);
            assert_eq!(cloned_tree.size(), 2);
        }

        #[test]
        fn it_holds_on_to_its_capacity_before_splitting() {
            let mut tree = KdTree::new_static();
            let capacity = 2_usize.pow(4);
            for _ in 0..capacity {
                let (pos, data) = random_point();
                tree.add(&pos, data).unwrap();
            }
            assert_eq!(tree.size, capacity);
            assert_eq!(tree.size(), capacity);
            assert!(tree.left.is_none() && tree.right.is_none());
            {
                let (pos, data) = random_point();
                tree.add(&pos, data).unwrap();
            }
            assert_eq!(tree.size, capacity + 1);
            assert_eq!(tree.size(), capacity + 1);
            assert!(tree.left.is_some() && tree.right.is_some());
        }

        #[test]
        fn no_items_can_be_added_to_a_zero_capacity_kdtree() {
            let mut tree = KdTree::with_capacity_static(0);
            let (pos, data) = random_point();
            let res = tree.add(&pos, data);
            assert!(res.is_err());
        }
    }

    mod dvector {
        extern crate rand;

        use nalgebra::DVector;

        use super::super::KdTree;

        const DIMS: usize = 2;

        fn random_point() -> (DVector<f64>, i32) {
            (DVector::new_random(DIMS), rand::random())
        }

        #[test]
        fn it_has_default_capacity() {
            let tree: KdTree<f64, i32, _> = KdTree::new_dynamic(DIMS);
            assert_eq!(tree.capacity, 2_usize.pow(4));
        }

        #[test]
        fn it_can_be_cloned() {
            let mut tree = KdTree::new_dynamic(DIMS);
            let (pos, data) = random_point();
            tree.add(&pos, data).unwrap();
            let mut cloned_tree = tree.clone();
            cloned_tree.add(&pos, data).unwrap();
            assert_eq!(tree.size(), 1);
            assert_eq!(cloned_tree.size(), 2);
        }

        #[test]
        fn it_holds_on_to_its_capacity_before_splitting() {
            let mut tree = KdTree::new_dynamic(DIMS);
            let capacity = 2_usize.pow(4);
            for _ in 0..capacity {
                let (pos, data) = random_point();
                tree.add(&pos, data).unwrap();
            }
            assert_eq!(tree.size, capacity);
            assert_eq!(tree.size(), capacity);
            assert!(tree.left.is_none() && tree.right.is_none());
            {
                let (pos, data) = random_point();
                tree.add(&pos, data).unwrap();
            }
            assert_eq!(tree.size, capacity + 1);
            assert_eq!(tree.size(), capacity + 1);
            assert!(tree.left.is_some() && tree.right.is_some());
        }

        #[test]
        fn no_items_can_be_added_to_a_zero_capacity_kdtree() {
            let mut tree = KdTree::with_capacity_dynamic(DIMS, 0);
            let (pos, data) = random_point();
            let res = tree.add(&pos, data);
            assert!(res.is_err());
        }
    }
}
