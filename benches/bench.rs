#![feature(test)]
extern crate kdtree;
extern crate nalgebra;
extern crate rand;
extern crate test;

use kdtree::norm::EuclideanNormSquared;
use kdtree::KdTree;
use nalgebra::{DVector, Vector3};
use test::Bencher;

fn rand_data_static() -> (Vector3<f64>, f64) {
    (Vector3::new_random(), rand::random())
}

#[bench]
fn bench_add_to_kdtree_with_1k_3d_points_static(b: &mut Bencher) {
    let len = 1000usize;
    let point = rand_data_static();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity_static(16);
    for _ in 0..len {
        points.push(rand_data_static());
    }
    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }
    b.iter(|| kdtree.add(&point.0, point.1).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_1k_3d_points_static(b: &mut Bencher) {
    let len = 1000usize;
    let point = rand_data_static();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity_static(16);
    for _ in 0..len {
        points.push(rand_data_static());
    }
    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }
    b.iter(|| kdtree.nearest(&point.0, 8, &EuclideanNormSquared).unwrap());
}

fn rand_data_dynamic() -> (DVector<f64>, f64) {
    (DVector::new_random(3), rand::random())
}

#[bench]
fn bench_add_to_kdtree_with_1k_3d_points_dynamic(b: &mut Bencher) {
    let len = 1000usize;
    let point = rand_data_dynamic();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity_dynamic(3, 16);
    for _ in 0..len {
        points.push(rand_data_dynamic());
    }
    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }
    b.iter(|| kdtree.add(&point.0, point.1).unwrap());
}

#[bench]
fn bench_nearest_from_kdtree_with_1k_3d_points_dynamic(b: &mut Bencher) {
    let len = 1000usize;
    let point = rand_data_dynamic();
    let mut points = vec![];
    let mut kdtree = KdTree::with_capacity_dynamic(3, 16);
    for _ in 0..len {
        points.push(rand_data_dynamic());
    }
    for i in 0..points.len() {
        kdtree.add(&points[i].0, points[i].1).unwrap();
    }
    b.iter(|| kdtree.nearest(&point.0, 8, &EuclideanNormSquared).unwrap());
}
